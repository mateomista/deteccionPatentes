"""Constantes y configuración del pipeline ALPR.

Hay tres tipos de valores acá, y la separación es la base de todo el esquema
entrenamiento / calibración / evaluación:

  1. CONSTANTES DE ALGORITMO (este módulo, nivel módulo): geometría de la chapa,
     parámetros de Sauvola/CLAHE, regex de formato, etc. No se barren en la
     calibración; si querés tocarlas, es un cambio de código versionado.

  2. `Thresholds`: los 8 umbrales [SIN CALIBRAR] del pipeline original. Son
     inmutables y se pasan al pipeline por constructor (antes eran globales que
     se pisaban con `global` desde la CLI). Es lo que barre `alpr.calibrate`.

  3. `PipelineOptions`: opciones de ejecución (rutas, fp16, batch, modo estricto).
"""
import json
import re
from dataclasses import dataclass, asdict, replace, fields
from typing import Optional

# ------------------------------------------------------------------------------
# Parámetros de operación (NO se calibran con la grilla de umbrales)
# ------------------------------------------------------------------------------
OCR_ZONE_START_RATIO = 0.40
OCR_INTERVAL_FRAMES = 5
TRACK_TTL_FRAMES = 60
FLUSH_UNFINALIZED_DEFAULT = True

# ------------------------------------------------------------------------------
# Recuperación / finalización: razones registradas en cada detección
# ------------------------------------------------------------------------------
CONSENSUS_REASON = "consensus"
RECOVERED_REASONS = ("track_lost", "end_of_video")
FINALIZATION_REASONS = (CONSENSUS_REASON,) + RECOVERED_REASONS

# Focus mask
# ------------------------------------------------------------------------------
# [FIX-v3 #2] El rango de aspecto depende de la clase de vehículo. Las chapas de
# auto son apaisadas (la Mercosur de automóvil es de 400x130 mm, aspecto ~3.1).
# Las de moto son de DOS líneas y aproximadamente cuadradas, así que con un piso
# de 1.4 el pipeline las descartaba silenciosamente antes de llegar al OCR.
# NOTA: no pude confirmar los milímetros exactos de la placa de motovehículo en
# la normativa; el piso de 0.9 es deliberadamente permisivo. Calibralo con tus
# propios recortes de moto antes de publicar métricas.
PLATE_GEOMETRY = {
    # clase: (min_w, min_h, aspect_min, aspect_max)
    "motorcycle": (36, 20, 0.9, 3.0),
    "default":    (48, 16, 1.4, 5.5),
}
MIN_GRAY_STD = 12.0   # desvío de gris mínimo: región uniforme => no hay texto

# Focus mask temprana sobre el recorte del VEHÍCULO: por debajo de esto no puede
# contener una chapa legible, así que se evita también el forward de RT-DETR.
MIN_VEHICLE_CROP_WIDTH = 64
MIN_VEHICLE_CROP_HEIGHT = 48
# ------------------------------------------------------------------------------
# [BATCH-#2] Inferencia batcheada de RT-DETR
# ------------------------------------------------------------------------------
# Presupuesto de VRAM en la RTX 3080 (10 GB): RF-DETR Medium + RT-DETR +
# EasyOCR residentes, más el pico de activaciones del lote. Con pixel_values
# de ~640x640, el tensor de entrada es ~4.6 MB/crop (fp32): las activaciones
# dominan. 8 es conservador; ante OOM el backoff degrada solo, sin morir.
DEFAULT_RTDETR_BATCH = 8

# ------------------------------------------------------------------------------
# Sauvola: T = mean * (1 + k * (std / R - 1))
# ------------------------------------------------------------------------------
SAUVOLA_WINDOW = 25   # ventana base en px, referida al crop SIN ampliar
SAUVOLA_K = 0.20
SAUVOLA_R = 128.0

BILATERAL_DIAMETER = 5
BILATERAL_SIGMA_COLOR = 30.0
BILATERAL_SIGMA_SPACE = 30.0

# [FIX-v5 #5] Polaridad: si la MEDIANA del gris (el fondo domina el histograma
# de una chapa) es menor que esto, la chapa es de fondo oscuro y se invierte.
POLARITY_MEDIAN_THRESHOLD = 127.0

# ------------------------------------------------------------------------------
# Limpieza de contornos
# ------------------------------------------------------------------------------
# [FIX-v3 #9] El umbral de área es relativo al recorte (con piso absoluto), para
# que no cambie de significado cuando el crop se amplía x2 o x4.
MIN_CONTOUR_AREA_ABS = 15
MIN_CONTOUR_AREA_RATIO = 0.0006
BORDER_MARGIN = 2
MIN_CHAR_HEIGHT_RATIO = 0.5   # un componente alto pegado al borde es un carácter cortado

# [FIX-v3 #1] Relleno de huecos internos ACOTADO. Rellenar todo hueco interno
# tapaba el ojo de O, 0, D, Q, 8, B: el pipeline destruía exactamente la
# información que el prewarp intenta preservar (O vs Q). Ahora solo se tapan
# huecos diminutos, que son ruido dentro del trazo, no contraformas del glifo.
HOLE_MAX_AREA_RATIO = 0.002    # respecto del área total del recorte
HOLE_MAX_HEIGHT_RATIO = 0.25   # el ojo de un "0" ocupa bastante más que esto

# ------------------------------------------------------------------------------
# Prewarp
# ------------------------------------------------------------------------------
PREWARP_MIN_ANGLE = 0.5
PREWARP_MAX_ANGLE = 20.0

# ------------------------------------------------------------------------------
# OCR
# ------------------------------------------------------------------------------
OCR_MIN_HEIGHT = 48
OCR_UPSCALE_MAX = 4.0
OCR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
OCR_TEXT_THRESHOLD = 0.5
OCR_LOW_TEXT = 0.3
OCR_SECOND_PASS_CONF = 0.60

# Fusión de fragmentos de EasyOCR
OCR_MIN_FRAGMENT_CONF = 0.10   # fragmentos por debajo de esto se ignoran
OCR_MIN_HEIGHT_RATIO = 0.55    # [FIX-v3 #3] descarta el texto chico de la banda
                               # azul (MERCOSUR / ARGENTINA) y los fantasmas
OCR_LINE_SPLIT_RATIO = 0.60    # separación vertical para considerar otra línea
OCR_MAX_MERGED_LEN = 8         # más que esto no es una patente: no fusionar

# ------------------------------------------------------------------------------
# [OCR-ADAPT] OCR adaptativo
# ------------------------------------------------------------------------------
# Tope DURO de intentos de EasyOCR por crop de patente (1 base + 3 variantes).
OCR_MAX_ATTEMPTS = 4
# Un candidato con formato válido pero confianza por debajo de esto se rechaza
# "por confianza". Igual al piso del consenso: lo que no llegaría a finalizar
# un track no tiene sentido votarlo.
# Confianza a partir de la cual un candidato válido corta la escalera de
# intentos del OCR adaptativo (misma que usaba el OCR dual).
OCR_SUFFICIENT_CONF = OCR_SECOND_PASS_CONF

OCR_VARIANT_BASE = "base"
# Orden de rescate: de la más barata/probable a la más agresiva.
OCR_VARIANT_ORDER = ("clahe", "clahe_unsharp", "sauvola_strong")

# Parámetros de las variantes. ESTIMADOS: calibralos A/B con
# --ocr-max-attempts sobre tu propio video antes de publicar métricas.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (4, 4)        # la chapa es chica: tiles chicos
UNSHARP_SIGMA = 1.2             # sobre el gris ya ampliado (>= OCR_MIN_HEIGHT)
UNSHARP_AMOUNT = 1.0
SAUVOLA_STRONG_K = 0.34         # vs SAUVOLA_K=0.20 del intento base
SAUVOLA_STRONG_WINDOW_FACTOR = 1.5
SAUVOLA_STRONG_CLAHE_CLIP = 3.0

# ------------------------------------------------------------------------------
# Validación
# ------------------------------------------------------------------------------
# [FIX-v5 #7] Valor por defecto de --strict-plate.
# [POLICY] DOS MODOS, NO EQUIVALENTES:
#   strict=True  -> VALIDACIÓN ESTRICTA: solo LLLNNN / LLNNNLL / LNNNLLL
#                   (PLATE_REGEX_PATTERNS). Es la única que significa "formato de
#                   patente argentino válido". Modo de PRODUCCIÓN: ejecutar con
#                   --strict-plate.
#   strict=False -> VALIDACIÓN PERMISIVA DE DEBUG (DEBUG_PLATE_REGEX): cualquier
#                   cadena de 6-7 caracteres A-Z/0-9. NO equivale a formato
#                   argentino: "ABCDEFG" o "123456" pasan. Sirve para diagnosticar
#                   el OCR, no para producir resultados citables.
# El default queda en False para no cambiar el comportamiento actual; el pipeline
# avisa por log (_log_policy_banner) cuando corre en modo permisivo.
STRICT_PLATE_VALIDATION = False
PLATE_REGEX_PATTERNS = (
    re.compile(r'^[A-Z]{3}[0-9]{3}$'),          # LLLNNN  (Argentina formato anterior)
    re.compile(r'^[A-Z]{2}[0-9]{3}[A-Z]{2}$'),  # LLNNNLL (Mercosur automóvil)
    re.compile(r'^[A-Z]{1}[0-9]{3}[A-Z]{3}$'),  # LNNNLLL (Mercosur motocicleta)
)
# [FIX-v3 #7] text.isupper() devuelve False para "123456" (no hay caracteres con
# caja), así que rechazaba lecturas numéricas por el motivo equivocado. Regex.
# [POLICY] SOLO validación permisiva de DEBUG (strict=False). No es un formato oficial.
DEBUG_PLATE_REGEX = re.compile(r'^[A-Z0-9]{6,7}$')

# ------------------------------------------------------------------------------
# [EXPORT] Esquema de exportación
# ------------------------------------------------------------------------------
EXPORT_FIELDS = (
    "id_vehiculo",
    "tipo_vehiculo",
    "patente_validada",
    "confianza_ocr_promedio",
    "cantidad_votos",
    "timestamp_seconds",
    "frame_idx",
    "snapshot_path",
    "bounding_box",
    "finalization",
    "format_valid",
)


# ==============================================================================
# UMBRALES CALIBRABLES
# ==============================================================================
@dataclass(frozen=True)
class Thresholds:
    """Los umbrales [SIN CALIBRAR] del pipeline. Valores iniciales = los de V5.

    Se dividen en dos grupos, y la diferencia importa para el costo de calibrar:

      DETECTION_FIELDS: cambian QUÉ ven los modelos (RF-DETR / RT-DETR), así que
        cada valor distinto exige volver a correr los modelos sobre el video.
      DECISION_FIELDS: solo cambian cómo se decide a partir de las lecturas ya
        hechas (votos, consenso, recuperación). Se barren OFFLINE sobre las
        observaciones grabadas, sin tocar la GPU (ver alpr/replay.py).
    """
    confidence_vehicle: float = 0.45      # RF-DETR (vehículos)
    confidence_plate: float = 0.25        # RT-DETR (patentes); bajo a propósito
    min_votes: int = 2
    min_avg_ocr_conf: float = 0.20        # también es el piso por candidato OCR
    early_finalize_votes: int = 3
    early_finalize_conf: float = 0.90
    min_consensus_margin: float = 0.15
    flush_single_vote_conf: float = 0.50

    def __post_init__(self):
        for name in ("confidence_vehicle", "confidence_plate", "min_avg_ocr_conf",
                     "early_finalize_conf", "min_consensus_margin", "flush_single_vote_conf"):
            v = getattr(self, name)
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"Thresholds.{name} debe estar en [0, 1], recibí {v}")
        for name in ("min_votes", "early_finalize_votes"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"Thresholds.{name} debe ser >= 1")

    def to_dict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Umbrales desconocidos: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # best_thresholds.json de la calibración anida los umbrales bajo "thresholds".
        if isinstance(data, dict) and "thresholds" in data:
            data = data["thresholds"]
        return cls.from_dict(data)

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


DETECTION_FIELDS = ("confidence_vehicle", "confidence_plate")
DECISION_FIELDS = ("min_votes", "min_avg_ocr_conf", "early_finalize_votes",
                   "early_finalize_conf", "min_consensus_margin",
                   "flush_single_vote_conf")


@dataclass
class PipelineOptions:
    """Opciones de ejecución (nada de esto es un umbral de aceptación)."""
    plate_model: str = "outputs/rtdetr-ccpd/final_model"
    snapshots_dir: Optional[str] = "snapshots"   # None = no escribir snapshots
    max_frames: int = 0                          # 0 = video completo
    dedup_seconds: float = 30.0
    fp16: bool = True
    dual_ocr: bool = True                        # variantes de rescate del OCR
    ocr_max_attempts: int = OCR_MAX_ATTEMPTS
    disable_focus_mask: bool = False
    debug_crops: bool = False
    rtdetr_batch: int = DEFAULT_RTDETR_BATCH
    sequential_rtdetr: bool = False
    strict_plate: bool = STRICT_PLATE_VALIDATION
    flush_unfinalized: bool = FLUSH_UNFINALIZED_DEFAULT
    cudnn_benchmark: bool = False
    log_discards: bool = True                    # registrar qué se descarta y por qué
