"""
Proyecto ALPR/LPR UTN-GIBD & Jefatura Departamental de Policía
Script: PIPELINE UNIFICADO EN TIEMPO REAL - v3
Hardware objetivo: NVIDIA GeForce RTX 3080

=============================================================================
QUÉ CAMBIA RESPECTO DE LA v2 (revisión crítica de la propia v2)
=============================================================================
Los cambios están marcados en el código con [FIX-v3] (defecto real corregido)
y [MEJORA-v3] (agregado nuevo). Resumen:

  [FIX-v3 #1] RELLENO DE HUECOS INTERNOS. La v2 (y las dos versiones previas)
      rellenaban TODO hueco interno que no tocara el borde. Eso tapa el ojo de
      O, 0, D, Q, 8, B, A, P y R: justamente los caracteres que el prewarp
      intenta desambiguar. Se rellenaban los contraformas de los glifos y se
      le entregaba a EasyOCR blobs sólidos indistinguibles entre sí. Ahora solo
      se rellenan huecos diminutos (ruido de sal y pimienta dentro del trazo),
      acotados por área relativa Y por altura relativa.

  [FIX-v3 #2] LAS MOTOS QUEDABAN FILTRADAS. El focus mask exigía aspecto >= 1.4
      para todo. Las chapas de moto son de dos líneas y casi cuadradas, así que
      el pipeline las descartaba antes del OCR pese a tener clase "motorcycle"
      y regex LNNNLLL. El rango de aspecto ahora depende de la clase de vehículo.

  [FIX-v3 #3] CHAPAS DE DOS LÍNEAS ROTAS EN EL MERGE. select_ocr_result solo
      fusionaba fragmentos de la MISMA línea horizontal, así que en una chapa de
      moto devolvía media patente. Ahora agrupa en líneas, las ordena de arriba
      hacia abajo y concatena, descartando por altura el texto chico de la banda
      azul (MERCOSUR / ARGENTINA).

  [FIX-v3 #4] inference_mode() CON POST-PROCESO AFUERA. Los tensores creados
      bajo inference_mode son "inference tensors" y usarlos fuera del contexto
      puede tirar RuntimeError. El forward y el post-proceso ahora ocurren
      dentro del mismo bloque y salen convertidos a numpy.

  [FIX-v3 #5] ESCALADO DESPUÉS DE BINARIZAR. Ampliar una imagen binaria con
      INTER_CUBIC devuelve valores intermedios (deja de ser binaria) y amplifica
      el escalonado. El escalado ahora se hace sobre el gris, ANTES del
      bilateral y de Sauvola, y la ventana de Sauvola se escala en proporción.

  [FIX-v3 #6] is_early ERA CÓDIGO MUERTO. Toda condición que cumplía is_early
      (3 votos, conf >= 0.90) ya cumplía is_standard (2 votos, conf >= 0.20),
      así que la rama nunca decidía nada. Ahora significa algo concreto: una
      lectura muy confiable y repetida finaliza salteando el margen de consenso.

  [FIX-v3 #7] text.isupper() EN LA VALIDACIÓN DEBUG. Devuelve False para una
      cadena sin letras, así que rechazaba lecturas puramente numéricas por un
      motivo equivocado. Reemplazado por un regex explícito.

  [FIX-v3 #8] LA MÉTRICA DE AHORRO ESTABA MAL CALCULADA. Un mismo contador
      mezclaba descartes a nivel vehículo (previos a RT-DETR) con descartes a
      nivel chapa (previos al OCR), y el denominador contaba PASADAS de OCR en
      vez de crops. Para una tesis eso es un número indefendible. Ahora hay
      contadores separados y el porcentaje se calcula sobre crops de chapa.

  [FIX-v3 #9] UMBRAL DE ÁREA ABSOLUTO. MIN_CONTOUR_AREA fijo cambia de
      significado según cuánto se amplió el crop. Ahora es relativo al área del
      recorte, con un piso absoluto.

  [FIX-v3 #10] EL PREWARP RECORTABA CARACTERES. Rotaba dentro del mismo lienzo,
      así que con ángulos grandes las esquinas (y los caracteres de los
      extremos) se salían del encuadre. Ahora el lienzo se expande.

  [MEJORA-v3 #11] --debug-crops: vuelca el recorte natural y el procesado de
      cada patente finalizada, para las figuras de la tesis y para diagnosticar
      por qué una lectura salió mal.

  [MEJORA-v3 #12] --max-frames para calibrar sobre un tramo corto.

PENDIENTE CONOCIDO: batcheo de RT-DETR (un forward por frame en lugar de uno
por vehículo). Es la mayor ganancia de throughput que queda sin tomar.

CALIBRACIÓN REQUERIDA: las constantes de aspecto y tamaño mínimo están
estimadas. Verificalas contra tu propio dataset antes de dar números finales.
=============================================================================
"""
# =============================================================================
# [BATCH-v4] pipeline_batch_dev.py — QUÉ CAMBIA RESPECTO DE pipelineV3.py
#
#  [BATCH-#1] Cierra el "PENDIENTE CONOCIDO" del header v3: RT-DETR de
#      patentes corre en dos fases por frame. FASE A (_collect_ocr_jobs)
#      hace toda la lógica barata en CPU (máquina de estados ROI, intervalo
#      de OCR, focus mask temprana de vehículo) y encola OcrJobs; FASE B
#      (_run_ocr_jobs) ejecuta UN forward de RT-DETR por lote de recortes.
#      Decisiones, contadores (FIX-v3 #8), votos y validación: idénticos.
#  [BATCH-#2] Chunking con backoff de OOM: el lote se parte en --rtdetr-batch
#      (default 8) y ante CUDA OOM se reduce a la mitad y se reintenta, para
#      convivir con RF-DETR y EasyOCR en los 10 GB de la RTX 3080.
#  [BATCH-#3] Se restaura _detect_plate_box (secuencial, idéntico a V3). Sin
#      él, _process_plate_ocr revientaba con AttributeError; hoy además
#      habilita --sequential-rtdetr para benchmark A/B batch vs secuencial
#      sobre el MISMO video y el MISMO árbol de decisión.
#  [BATCH-#4] Telemetría de batch (forwards, crops/forward, pico de VRAM) y
#      TF32 + cudnn.benchmark para Ampere.
# =============================================================================
# [REV-v5] REVISIÓN DE pipeline_batch_dev.py (BATCH-v4). Marcas: [FIX-v5].
#
#  [FIX-v5 #1] EL ARCHIVO NO COMPILABA. (a) _detect_plate_box_batch estaba
#      indentado 8 espacios: quedaba anidado dentro de _detect_plate_box,
#      después del return. (b) el final de _run_ocr_jobs traía un bloque
#      `finally:` sin `try:` y con marcas "[cite: 1]" (resto de pegar un
#      fragmento de run()). Ambos corregidos; run() ya hacía lo que ese
#      fragmento intentaba hacer.
#  [FIX-v5 #2] _log_stats estaba definido DOS veces (la de FIX-v3 y la de
#      BATCH). Python se queda con la última, pero la primera confunde y se
#      puede editar por error. Queda una sola.
#  [FIX-v5 #3] BACKOFF DE OOM. (a) Un RuntimeError que NO era OOM se tragaba
#      y el código seguía con `boxes` sin definir (o con los del chunk
#      anterior, asignando cajas a los vehículos equivocados). Ahora se
#      re-lanza. (b) El lote reducido se recuerda entre frames: antes cada
#      frame reintentaba el lote completo, volvía a fallar y a vaciar caché.
#      (c) El chunking estaba duplicado (_run_ocr_jobs y _..._batch_safe).
#  [FIX-v5 #4] TRACKS SIN CONSENSO SE PERDÍAN EN SILENCIO. Un vehículo cuya
#      patente se leyó una sola vez, o con margen < MIN_CONSENSUS_MARGIN,
#      nunca se finalizaba: _purge_stale_tracks lo borraba y al terminar el
#      video no se vaciaba nada. Ahora se vacían al perder el track y al fin
#      del video, con criterio conservador (ver _flush_track) y con el
#      registro marcado en "finalization" y "format_valid" para poder
#      filtrarlos. Se desactiva con --no-flush-unfinalized.
#  [FIX-v5 #5] POLARIDAD. La inversión "chapa vieja fondo negro" se decidía
#      DESPUÉS de Sauvola con la media de la binaria. Sauvola manda las
#      regiones uniformes a blanco (std=0 => T = 0.8*media < píxel), así que
#      una chapa negra sale casi toda blanca, la media da alta y la inversión
#      no se dispara: el texto se perdía. Ahora la polaridad se normaliza
#      sobre el GRIS (mediana < 127 => invertir) antes de binarizar, y también
#      en la segunda pasada en gris.
#  [FIX-v5 #6] cudnn.benchmark=True con EasyOCR es contraproducente: CRAFT y
#      el reconocedor reciben shapes distintas en cada crop y el autotuner se
#      dispara con cada shape nueva. Ahora es opt-in (--cudnn-benchmark) para
#      que lo compares A/B en tu video.
#  [FIX-v5 #7] STRICT_PLATE_VALIDATION estaba fijo en False en el código.
#      Ahora es --strict-plate, y cada registro lleva "format_valid" (regex
#      oficial) aunque se corra en modo permisivo.
#  [FIX-v5 #8] smart_correct_plate con 7 caracteres desempataba siempre a
#      favor de LLNNNLL (auto). Ahora, si hay empate, usa la clase del
#      vehículo (moto => LNNNLLL).
#  [FIX-v5 #9] select_ocr_result usaba `in` (== sobre tuplas con listas de
#      coordenadas). Ahora compara por identidad.
#  [FIX-v5 #10] _clean_binary_contours hacía labels == i por componente
#      (O(componentes x píxeles)). Ahora usa una tabla de búsqueda: mismo
#      resultado, mucho más barato.
#  [FIX-v5 #11] El log del mapeo de clases se emitía al importar, antes de
#      basicConfig, y nunca se veía. Movido a _load_models. Además se
#      protege tracked.tracker_id is None.
# =============================================================================
# [OCR-ADAPT] OCR ADAPTATIVO (máx. 4 intentos por crop de patente).
# Alcance: SOLO el subsistema OCR y sus métricas. RF-DETR, RT-DETR, ByteTrack,
# el consenso temporal y la exportación no se tocan en este bloque.
#
#  [OCR-ADAPT #1] _read_plate_adaptive reemplaza a _read_plate_text. Intento 1
#      ("base"): pipeline actual (prewarp + gris + polaridad + bilateral +
#      Sauvola + limpieza). Si no sale una lectura válida con confianza
#      suficiente se generan, de a una y en este orden, variantes
#      INDEPENDIENTES: "clahe", "clahe_unsharp", "sauvola_strong".
#  [OCR-ADAPT #2] Independencia: todas las variantes parten de la misma base
#      geométrica (el plate_crop natural + el prewarp, que es una corrección
#      de ángulo y no un filtro) y cada una arma su propia cadena de filtros.
#      Ninguna recibe como entrada la salida de otra.
#  [OCR-ADAPT #3] Corte temprano: se deja de intentar apenas hay un candidato
#      válido (smart_correct_plate + is_valid_plate) con confianza >=
#      OCR_SUFFICIENT_CONF. Un candidato válido pero de confianza baja
#      también habilita los intentos siguientes (es el mismo criterio que
#      usaba el OCR dual: reintentar si la confianza no alcanza).
#  [OCR-ADAPT #4] UN solo candidato final por crop/frame (válido, con
#      preferencia por formato oficial y luego mayor confianza) y por lo tanto
#      UN solo voto. Si ningún intento es válido no se devuelve nada: el
#      llamador no vota, no toca el track y el próximo intento ocurre en un
#      frame posterior (OCR_INTERVAL_FRAMES, como antes).
#  [OCR-ADAPT #5] Se retira la pasada "dual" separada (gris + polaridad, sin
#      contraste): quedaba subsumida por la variante "clahe" y habría sido un
#      quinto intento. --dual-ocr ahora habilita las variantes de rescate;
#      --no-dual-ocr deja solo el intento 1. --ocr-max-attempts (1..4) permite
#      acotar el presupuesto para comparaciones A/B.
#  [OCR-ADAPT #6] Contadores separados: crops procesados, intentos totales,
#      intentos por variante, lecturas válidas, rechazadas por formato,
#      rechazadas por confianza (más: sin texto y variante ganadora por crop,
#      que cierran la cuenta: intentos = válidas + rechazos + sin texto).
# =============================================================================
# [EXPORT] EXPORTACIÓN DE RESULTADOS. Alcance: _finalize_track y la salida
# (JSON final y JSONL incremental). Lógica de detección/OCR/consenso intacta.
#
#  [EXPORT #1] Esquema de cada detección (lista de objetos, no un dict):
#      id_vehiculo, tipo_vehiculo, patente_validada, confianza_ocr_promedio,
#      cantidad_votos, timestamp_seconds, frame_idx, snapshot_path,
#      bounding_box, finalization, format_valid. Sin marca/modelo/color.
#  [EXPORT #2] Integridad de evidencia: PlateEvidence guarda el texto al que
#      pertenece; _finalize_track rechaza (y loguea) cualquier registro cuya
#      evidencia sea de otra patente, tenga frame_idx < 0, carezca de recorte
#      o bbox. Se eliminó el fallback "timestamp = 0.0" que enmascaraba
#      evidencia ausente.
#  [EXPORT #3] La patente exportada debe pasar is_valid_plate con la
#      validación configurada (--strict-plate).
#  [EXPORT #4] _finalize_track devuelve si realmente registró; el contador
#      tracks_flushed ya no cuenta duplicados ni registros rechazados.
#  [EXPORT #5] Auditoría final (_audit_detections) sobre json_log: campos
#      obligatorios, timestamp >= 0, frame_idx >= 0, bbox de 4 enteros,
#      patente válida y snapshot coherente con patente y frame.
# =============================================================================

import os
import re
import cv2
import json
import torch
import logging
import argparse
import numpy as np
import supervision as sv
import easyocr
import time  # [BATCH-#4] telemetría de FPS y duración
from enum import Enum
from collections import defaultdict
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from rfdetr import RFDETRMedium

# [FIX-DIAG] Antes se hardcodeaba VEHICLE_CLASS_NAMES = {2:"car",3:"motorcycle",
# 5:"bus",7:"truck"}, asumiendo la numeracion "estandar" de COCO-80. RF-DETR
# tiene su PROPIA lista oficial de clases (rfdetr.util.coco_classes, o
# rfdetr.assets.coco_classes segun la version del paquete) y hay casos
# documentados de desfasajes de indexado de a uno entre versiones/checkpoints
# (ver https://github.com/roboflow/rf-detr/issues/330). Si el indexado real
# no coincide con el asumido, un auto puede llegar con class_id=3 en vez de 2
# y quedar etiquetado "motorcycle": el focus mask del pipeline v3 le aplicaria
# entonces la geometria de PATENTE DE MOTO (casi cuadrada) a una patente de
# auto (alargada), rechazandola casi siempre. Por eso el mapeo se arma
# preguntandole a la propia libreria, buscando por NOMBRE, nunca por id fijo.
try:
    from rfdetr.util.coco_classes import COCO_CLASSES
except ImportError:
    from rfdetr.assets.coco_classes import COCO_CLASSES

_WANTED_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}
# COCO_CLASSES puede ser una lista posicional (id = indice) o un diccionario
# {id: nombre} con los ids "sparse" originales de COCO (1-90, con huecos) --
# depende de la version de rfdetr. enumerate() solo tiene sentido para el
# caso lista; sobre un dict iteraria las CLAVES, nunca los nombres, y el
# mapeo quedaria vacio (como paso). Se cubre explicitamente cada caso.
if isinstance(COCO_CLASSES, dict):
    _id_name_pairs = COCO_CLASSES.items()
else:
    _id_name_pairs = enumerate(COCO_CLASSES)

VEHICLE_CLASS_NAMES = {
    class_id: name for class_id, name in _id_name_pairs
    if name in _WANTED_VEHICLE_NAMES
}
if set(VEHICLE_CLASS_NAMES.values()) != _WANTED_VEHICLE_NAMES:
    logging.warning(
        "No se encontraron todas las clases de vehiculo esperadas en "
        f"COCO_CLASSES de rfdetr. Encontradas: {VEHICLE_CLASS_NAMES}. "
        "Revisa rfdetr.util.coco_classes.COCO_CLASSES a mano."
    )
# [FIX-v5 #11] El logging.info del mapeo resuelto se movió a _load_models:
# acá corre antes de basicConfig() y el nivel por defecto (WARNING) lo tiraba.

# ==============================================================================
# [POLICY] UMBRALES DE ACEPTACIÓN Y FINALIZACIÓN
# ==============================================================================
# Ningún valor de esta sección es FINAL. Son constantes configurables (se pueden
# sobreescribir por CLI, ver parse_args/_apply_threshold_overrides) y los marcados
# "[SIN CALIBRAR]" son valores iniciales ESTIMADOS, deliberadamente permisivos,
# que NO fueron validados contra el dataset/video real. No se deben citar como
# resultados definitivos hasta calibrarlos (curva precisión/recall contra
# verdad-terreno etiquetada). El pipeline los imprime al arrancar
# (_log_policy_banner) y UNCALIBRATED_THRESHOLDS lista cuáles son.
#
# Estos NO se calibran con esta sección: OCR_ZONE_START_RATIO, OCR_INTERVAL_FRAMES,
# TRACK_TTL_FRAMES (parámetros de operación, no de aceptación).
CONFIDENCE_VEHICLE = 0.45      # [SIN CALIBRAR] umbral de RF-DETR (vehículos)
CONFIDENCE_PLATE = 0.25        # [SIN CALIBRAR] umbral de RT-DETR (patentes); bajo a propósito
OCR_ZONE_START_RATIO = 0.40
OCR_INTERVAL_FRAMES = 5
TRACK_TTL_FRAMES = 60

# ------------------------------------------------------------------------------
# Criterios de finalización POR CONSENSO (finalization = "consensus")
# ------------------------------------------------------------------------------
MIN_VOTES_TO_FINALIZE = 2             # [SIN CALIBRAR]
MIN_AVG_OCR_CONFIDENCE = 0.20         # [SIN CALIBRAR] piso reducido; también fija OCR_CANDIDATE_MIN_CONF
# [FIX-v3 #6] Ahora la vía rápida tiene un significado propio: saltear el margen
# de consenso. Antes era un subconjunto de la condición estándar y no decidía
# nada. Sirve para el caso "el track leyó la misma patente 3 veces con 0.9+ pero
# arrastra un competidor viejo que le come el margen".
EARLY_FINALIZE_VOTES = 3              # [SIN CALIBRAR]
EARLY_FINALIZE_CONFIDENCE = 0.90      # [SIN CALIBRAR]
MIN_CONSENSUS_MARGIN = 0.15           # [SIN CALIBRAR]

# ------------------------------------------------------------------------------
# Recuperación de tracks SIN consenso (finalization = "track_lost" | "end_of_video")
# ------------------------------------------------------------------------------
# [FIX-v5 #4] Vaciado de tracks que nunca alcanzaron consenso (track perdido o
# fin de video). Con >= MIN_VOTES_TO_FINALIZE votos se acepta el ganador aunque
# el margen no llegue. Con UNA sola lectura solo se acepta si cumple el formato
# oficial Y supera FLUSH_SINGLE_VOTE_MIN_CONF.
# [POLICY] Estas detecciones son RECUPERADAS: su calidad NO es equivalente a la de
# una detección confirmada por consenso. Se distinguen SIEMPRE por el campo
# "finalization" (en JSON/JSONL), y el análisis de resultados debe reportarlas
# por separado o excluirlas (ver CONSENSUS_REASON / RECOVERED_REASONS).
FLUSH_UNFINALIZED_DEFAULT = True
FLUSH_SINGLE_VOTE_MIN_CONF = 0.50     # [SIN CALIBRAR]

# Umbrales que requieren calibración sobre el dataset/video real (solo
# informativo: se usa para el banner de arranque, no altera la lógica).
UNCALIBRATED_THRESHOLDS = (
    "CONFIDENCE_VEHICLE",
    "CONFIDENCE_PLATE",
    "MIN_VOTES_TO_FINALIZE",
    "MIN_AVG_OCR_CONFIDENCE",
    "EARLY_FINALIZE_VOTES",
    "EARLY_FINALIZE_CONFIDENCE",
    "MIN_CONSENSUS_MARGIN",
    "FLUSH_SINGLE_VOTE_MIN_CONF",
)

# ------------------------------------------------------------------------------
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
OCR_CANDIDATE_MIN_CONF = MIN_AVG_OCR_CONFIDENCE   # [SIN CALIBRAR] acoplado a MIN_AVG_OCR_CONFIDENCE
# Confianza a partir de la cual un candidato válido corta la escalera de
# intentos (mismo valor que usaba el OCR dual para decidir la segunda pasada).
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
# [POLICY] Calidad de la detección según cómo se cerró el track:
#   CONSENSUS_REASON   -> confirmada por consenso (check_finalization).
#   RECOVERED_REASONS  -> recuperada al perder el track o terminar el video SIN
#                         consenso (_flush_track): evidencia más débil.
# FINALIZATION_REASONS es la unión y no cambió (mismos valores que la auditoría
# ya validaba).
CONSENSUS_REASON = "consensus"
RECOVERED_REASONS = ("track_lost", "end_of_video")
FINALIZATION_REASONS = (CONSENSUS_REASON,) + RECOVERED_REASONS


# ==============================================================================
# ESTRUCTURAS DE DATOS Y ENUMERACIONES
# ==============================================================================
class TrackState(Enum):
    NEW = "NEW"
    OCR_ACTIVE = "OCR_ACTIVE"
    FINALIZED = "FINALIZED"


class PlateEvidence:
    """Mejor evidencia visual asociada específicamente a UN texto de patente."""
    def __init__(self):
        self.best_conf = 0.0
        self.text = None            # [EXPORT #2] patente a la que pertenece esta evidencia
        self.crop = None            # recorte NATURAL (lo que vio la cámara)
        self.processed = None       # [MEJORA-v3 #11] lo que vio el OCR
        self.frame = -1
        self.bbox = None


class VehicleTrack:
    """Centraliza todo el estado y la evidencia asociada a un track_id."""
    def __init__(self, track_id, vehicle_class_name):
        self.track_id = track_id
        self.vehicle_class_name = vehicle_class_name
        self.state = TrackState.NEW
        self.last_seen_frame = -1
        self.last_ocr_frame = -999

        self.votes = {}        # {plate_text: sum_confidence}
        self.vote_counts = {}  # {plate_text: count}
        self.evidence = defaultdict(PlateEvidence)

        self.best_plate_text = None

    def add_vote(self, text, confidence, frame_idx, plate_crop, processed_crop, vehicle_bbox):
        self.votes[text] = self.votes.get(text, 0.0) + float(confidence)
        self.vote_counts[text] = self.vote_counts.get(text, 0) + 1

        ev = self.evidence[text]
        if confidence > ev.best_conf:
            ev.best_conf = float(confidence)
            ev.text = text      # [EXPORT #2] la evidencia declara a qué patente pertenece
            ev.crop = plate_crop.copy() if plate_crop is not None else None
            ev.processed = processed_crop.copy() if processed_crop is not None else None
            ev.frame = frame_idx
            ev.bbox = vehicle_bbox.copy() if vehicle_bbox is not None else None

    def get_consensus_plate(self):
        """Patente ganadora por sumatoria total de confianza."""
        if not self.votes:
            return None, 0.0, 0
        best_plate = max(self.votes, key=self.votes.get)
        count = self.vote_counts[best_plate]
        avg_conf = self.votes[best_plate] / count
        return best_plate, avg_conf, count

    def check_finalization(self):
        if self.state == TrackState.FINALIZED:
            return True
        best_plate, avg_conf, count = self.get_consensus_plate()
        if not best_plate:
            return False

        best_score = self.votes[best_plate]
        margin = 1.0
        if len(self.votes) > 1:
            sorted_plates = sorted(self.votes.keys(), key=self.votes.get, reverse=True)
            second_score = self.votes[sorted_plates[1]]
            margin = (best_score - second_score) / best_score if best_score > 0 else 0.0

        # [FIX-v3 #6] La vía rápida ahora saltea el margen; la estándar lo exige.
        if count >= EARLY_FINALIZE_VOTES and avg_conf >= EARLY_FINALIZE_CONFIDENCE:
            return True

        is_standard = count >= MIN_VOTES_TO_FINALIZE and avg_conf >= MIN_AVG_OCR_CONFIDENCE
        return bool(is_standard and margin >= MIN_CONSENSUS_MARGIN)


# ==============================================================================
# UTILIDADES DE IMAGEN
# ==============================================================================
def _ensure_gray(img):
    """Guarda de ndim: cvtColor(BGR2GRAY) explota si la imagen ya es 2D."""
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _ocr_scale_factor(height):
    """Factor de ampliación necesario para llegar a OCR_MIN_HEIGHT, con tope."""
    if height <= 0 or height >= OCR_MIN_HEIGHT:
        return 1.0
    return min(OCR_UPSCALE_MAX, float(OCR_MIN_HEIGHT) / float(height))


def _resize(img, scale):
    if img is None or img.size == 0 or scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img,
                      (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_CUBIC)


def _normalize_polarity(gray):
    """[FIX-v5 #5] Lleva la chapa a texto oscuro sobre fondo claro, sobre el GRIS.

    En una chapa el fondo ocupa la mayor parte de los píxeles, así que la
    MEDIANA del gris identifica la polaridad de forma robusta: mediana baja =>
    chapa de fondo oscuro con letras claras => se invierte.

    Esto tiene que ocurrir ANTES de binarizar. Sauvola manda las regiones
    uniformes a blanco (con std=0, T = media*(1-k) < píxel), por lo que una
    chapa negra sale mayormente blanca de la binarización y la media de la
    binaria ya no dice nada sobre su polaridad original.
    """
    if gray is None or gray.size == 0:
        return gray
    if float(np.median(gray)) < POLARITY_MEDIAN_THRESHOLD:
        return cv2.bitwise_not(gray)
    return gray


# ==============================================================================
# PRE/POST PROCESAMIENTO (mitigaciones de la tesis)
# ==============================================================================
def _sauvola_threshold(gray, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K, r=SAUVOLA_R):
    """Binarización local de Sauvola: T = mean * (1 + k * (std / R - 1)).

    A diferencia de un umbral global (Otsu), calcula un umbral por píxel con las
    estadísticas de su vecindad, así que resiste sombras proyectadas y
    degradados suaves de iluminación sobre la chapa.
    """
    gray = _ensure_gray(gray)
    window_size = int(window_size)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1  # la ventana de la media móvil debe tener centro

    gray_f = gray.astype(np.float32)
    ksize = (window_size, window_size)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=ksize, normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=ksize, normalize=True,
                            borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    threshold = mean * (1.0 + k * ((std / r) - 1.0))
    return np.where(gray_f > threshold, 255, 0).astype(np.uint8)


def _prewarp_plate(plate_crop):
    """Endereza virtualmente la chapa antes del OCR (mitiga O vs Q).

    El ángulo sale del lado más largo del rectángulo mínimo (boxPoints +
    arctan2) en vez de rect[2]: la convención de ángulos de cv2.minAreaRect
    cambió en OpenCV 4.5 (hoy devuelve [0, 90)), así que el clásico
    `if angle < -45: angle = 90 + angle` puede reportar 88° para una chapa
    inclinada 2° y el filtro de rango la descarta sin que nadie se entere.

    [FIX-v3 #10] El lienzo se expande al bounding box de la rotación. Antes se
    rotaba dentro del mismo alto/ancho, de modo que con ángulos cercanos a los
    20° permitidos los caracteres de los extremos se salían del encuadre: el
    prewarp arreglaba la inclinación y de paso borraba la primera letra.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop
    try:
        gray = _ensure_gray(plate_crop)
        edges = cv2.Canny(gray, 50, 150)
        coords = cv2.findNonZero(edges)
        if coords is None or len(coords) < 10:
            return plate_crop

        box = cv2.boxPoints(cv2.minAreaRect(coords))
        d01 = float(np.linalg.norm(box[1] - box[0]))
        d12 = float(np.linalg.norm(box[2] - box[1]))
        edge = (box[1] - box[0]) if d01 >= d12 else (box[2] - box[1])
        angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        if abs(angle) < PREWARP_MIN_ANGLE or abs(angle) > PREWARP_MAX_ANGLE:
            return plate_crop

        h, w = plate_crop.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Lienzo expandido para no cortar las esquinas.
        cos_a = abs(matrix[0, 0])
        sin_a = abs(matrix[0, 1])
        new_w = int(round(h * sin_a + w * cos_a))
        new_h = int(round(h * cos_a + w * sin_a))
        matrix[0, 2] += (new_w / 2.0) - cx
        matrix[1, 2] += (new_h / 2.0) - cy

        return cv2.warpAffine(plate_crop, matrix, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        logging.debug(f"[prewarp] Error recuperable, se usa el crop original: {e}")
        return plate_crop


def _has_plate_characteristics(crop, vehicle_class="default"):
    """Focus mask: filtro barato ANTES de gastar cómputo de OCR.

    Descarta regiones demasiado chicas (proxy de distancia extrema), con aspecto
    fuera del rango esperado, o uniformes (fondo del auto, sin texto). Acá nace
    el ahorro de cómputo: cuesta microsegundos y evita invocar EasyOCR, la etapa
    más cara por crop.

    [FIX-v3 #2] La geometría esperada depende de la clase de vehículo: las
    chapas de moto son de dos líneas y casi cuadradas, y con el rango único de
    la v2 (aspecto >= 1.4) quedaban descartadas antes del OCR.

    [DIAG] Devuelve (paso, motivo) en vez de solo bool, para poder loguear
    EXACTAMENTE por que se descarta un crop que RT-DETR si detecto. "motivo"
    es None cuando paso=True.
    """
    if crop is None or crop.size == 0 or crop.ndim < 2:
        return False, "crop_vacio"

    min_w, min_h, aspect_min, aspect_max = PLATE_GEOMETRY.get(
        vehicle_class, PLATE_GEOMETRY["default"])

    h, w = crop.shape[:2]
    if w < min_w or h < min_h:
        return False, f"tamano ({w}x{h} < minimo {min_w}x{min_h})"

    aspect = w / max(h, 1)
    if aspect < aspect_min or aspect > aspect_max:
        return False, f"aspecto ({aspect:.2f} fuera de [{aspect_min}, {aspect_max}])"

    gray = _ensure_gray(crop)
    std = float(np.std(gray))
    if std < MIN_GRAY_STD:
        return False, f"desvio_gris ({std:.1f} < minimo {MIN_GRAY_STD})"  # region uniforme => no hay texto

    return True, None


def _clean_binary_contours(binary):
    """Limpieza de contornos sobre la imagen binaria.

    Mitiga los ruidos propios de las chapas argentinas:
      1) Componentes con área despreciable: puntos sueltos y restos del texto
         fantasma gris MERCOSUR/ARGENTINA.
      2) Componentes que tocan el borde: marco de la chapa (que el OCR leía como
         "1" o "I") y sellos circulares RNPA de las chapas previas a 2016.
         Excepción: si el componente es alto respecto del crop y no atraviesa
         todo el ancho/alto, se conserva — un dígito pegado al borde por un
         recorte ajustado es un caso frecuente, no un marco.
      3) [FIX-v3 #1] Huecos internos: se rellenan SOLO los diminutos.

    Sobre el punto 3, que es el defecto más serio que arrastraban las tres
    versiones anteriores: rellenar todo hueco interno que no tocara el borde
    tapa la contraforma de O, 0, D, Q, 8, B, A, P y R. El resultado es que
    EasyOCR recibía blobs sólidos donde antes había caracteres distinguibles, y
    el sistema perdía justamente la distinción O vs Q que el prewarp existe para
    proteger. Ahora un hueco se rellena solo si es chico en área relativa Y bajo
    en altura relativa, que es el perfil del ruido de sal y pimienta dentro del
    trazo, no el de la contraforma de un glifo.

    [FIX-v5 #10] La reconstrucción de la máscara usa una tabla de búsqueda
    (keep[labels]) en vez de `labels == i` por componente: mismo resultado.

    Convención de entrada/salida: texto oscuro sobre fondo claro.
    """
    if binary is None or binary.size == 0:
        return binary
    try:
        binary = _ensure_gray(binary)
        fg = cv2.bitwise_not(binary)  # texto = 255, fondo = 0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        h, w = fg.shape

        # [FIX-v3 #9] Umbral relativo al recorte, con piso absoluto: un valor
        # fijo cambia de significado según cuánto se haya ampliado el crop.
        min_area = max(MIN_CONTOUR_AREA_ABS, int(MIN_CONTOUR_AREA_RATIO * h * w))
        keep = np.zeros(num, dtype=np.uint8)

        for i in range(1, num):  # 0 es el fondo
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]

            if area < min_area:
                continue

            touches_border = (
                x <= BORDER_MARGIN or y <= BORDER_MARGIN or
                (x + ww) >= (w - BORDER_MARGIN) or (y + hh) >= (h - BORDER_MARGIN)
            )
            if touches_border:
                height_ratio = hh / h if h > 0 else 0.0
                spans_full_width = ww >= (w - 2 * BORDER_MARGIN)
                spans_full_height = hh >= (h - 2 * BORDER_MARGIN) and ww < (0.5 * w)
                looks_like_frame_or_seal = spans_full_width or spans_full_height
                looks_like_character = (height_ratio >= MIN_CHAR_HEIGHT_RATIO
                                        and not looks_like_frame_or_seal)
                if not looks_like_character:
                    continue

            keep[i] = 255

        cleaned_fg = keep[labels]

        # Relleno acotado de huecos: solo ruido dentro del trazo.
        inv = cv2.bitwise_not(cleaned_fg)
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        max_hole_area = max(MIN_CONTOUR_AREA_ABS, int(HOLE_MAX_AREA_RATIO * h * w))
        max_hole_height = HOLE_MAX_HEIGHT_RATIO * h
        fill = np.zeros(num2, dtype=np.uint8)

        for i in range(1, num2):
            x = stats2[i, cv2.CC_STAT_LEFT]
            y = stats2[i, cv2.CC_STAT_TOP]
            ww = stats2[i, cv2.CC_STAT_WIDTH]
            hh = stats2[i, cv2.CC_STAT_HEIGHT]
            area = stats2[i, cv2.CC_STAT_AREA]

            is_internal = (x > BORDER_MARGIN and y > BORDER_MARGIN and
                           (x + ww) < (w - BORDER_MARGIN) and (y + hh) < (h - BORDER_MARGIN))
            if not is_internal:
                continue
            if area > max_hole_area or hh > max_hole_height:
                continue  # es la contraforma de un glifo: NO tocar
            fill[i] = 255

        cleaned_fg = cv2.bitwise_or(cleaned_fg, fill[labels2])

        return cv2.bitwise_not(cleaned_fg)
    except Exception as e:
        logging.warning(f"[clean] Limpieza de contornos falló, se usa la binarizada cruda: {e}")
        return binary


def _preprocess_plate_for_ocr(plate_crop):
    """Cadena de preprocesamiento que ve EasyOCR (NO la evidencia visual).

    [OCR-ADAPT] Es el intento 1 ("base") del OCR adaptativo; no cambió.

    Orden:
      1) [FIX-v3 #5] Escalado sobre el GRIS. La v2 ampliaba al final, sobre la
         imagen ya binarizada: INTER_CUBIC sobre una binaria devuelve valores
         intermedios (la imagen deja de ser binaria) y amplifica el escalonado
         de los bordes. Ampliar antes le da además a Sauvola más píxeles reales
         por trazo con los que trabajar.
      2) [FIX-v5 #5] Normalización de polaridad sobre el gris: las Mercosur son
         fondo blanco / letras negras y las viejas fondo negro / letras blancas.
         Se decide con la mediana del gris ANTES de binarizar (después de
         Sauvola ya no es confiable, ver _normalize_polarity).
      3) Filtro bilateral: aplana ruido sin destruir los bordes de los caracteres.
      4) Sauvola con la ventana escalada en la misma proporción, para que siga
         cubriendo la misma cantidad de trazo que a escala original.
      5) Limpieza de contornos.

    Devuelve binaria lista para EasyOCR, o None si el preprocesamiento falló.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    try:
        gray = _ensure_gray(plate_crop)

        scale = _ocr_scale_factor(gray.shape[0])
        gray = _resize(gray, scale)

        gray = _normalize_polarity(gray)

        gray = cv2.bilateralFilter(gray, BILATERAL_DIAMETER,
                                   BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

        window = int(round(SAUVOLA_WINDOW * scale))
        binary = _sauvola_threshold(gray, window, SAUVOLA_K, SAUVOLA_R)

        return _clean_binary_contours(binary)
    except Exception as e:
        logging.warning(f"[preproc] Preprocesamiento de patente falló: {e}")
        return None


# ------------------------------------------------------------------------------
# [OCR-ADAPT] Variantes de rescate. Cada una es una función PURA e
# INDEPENDIENTE: recibe la misma base geométrica (plate_crop natural, ya
# enderezado por el prewarp) y arma su propia cadena de filtros desde cero.
# Ninguna recibe la salida de otra, y ninguna modifica su entrada in situ.
# ------------------------------------------------------------------------------
def _gray_base_for_ocr(plate_crop):
    """Base común de las variantes: gris ampliado y con polaridad normalizada.

    Es el mismo tratamiento previo al filtrado que usa el intento base (escala
    + polaridad), de modo que las variantes difieren SOLO en el filtrado.
    Devuelve (gris, escala) o (None, 1.0).
    """
    gray = _ensure_gray(plate_crop)
    if gray is None or gray.size == 0:
        return None, 1.0
    scale = _ocr_scale_factor(gray.shape[0])
    gray = _resize(gray, scale)
    return _normalize_polarity(gray), scale


def _variant_clahe(plate_crop):
    """Gris + CLAHE (normalización local de contraste)."""
    try:
        gray, _ = _gray_base_for_ocr(plate_crop)
        if gray is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        return clahe.apply(gray)
    except Exception as e:
        logging.debug(f"[ocr_variant:clahe] falló: {e}")
        return None


def _variant_clahe_unsharp(plate_crop):
    """Gris + CLAHE + Unsharp Mask (realza bordes de trazo borroso)."""
    try:
        gray, _ = _gray_base_for_ocr(plate_crop)
        if gray is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        eq = clahe.apply(gray)
        blur = cv2.GaussianBlur(eq, (0, 0), UNSHARP_SIGMA)
        return cv2.addWeighted(eq, 1.0 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0)
    except Exception as e:
        logging.debug(f"[ocr_variant:clahe_unsharp] falló: {e}")
        return None


def _variant_sauvola_strong(plate_crop):
    """Binarización Sauvola reforzada: CLAHE fuerte + suavizado + Sauvola con
    k y ventana mayores que el intento base + limpieza de contornos.

    Se diferencia del intento base en el pre-filtro (CLAHE + gaussiano en vez de
    bilateral) y en los parámetros de Sauvola, para cubrir chapas con
    iluminación despareja o trazo grueso donde el base falla.
    """
    try:
        gray, scale = _gray_base_for_ocr(plate_crop)
        if gray is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=SAUVOLA_STRONG_CLAHE_CLIP, tileGridSize=CLAHE_TILE_GRID)
        eq = cv2.GaussianBlur(clahe.apply(gray), (3, 3), 0)
        window = int(round(SAUVOLA_WINDOW * scale * SAUVOLA_STRONG_WINDOW_FACTOR))
        binary = _sauvola_threshold(eq, window, SAUVOLA_STRONG_K, SAUVOLA_R)
        return _clean_binary_contours(binary)
    except Exception as e:
        logging.debug(f"[ocr_variant:sauvola_strong] falló: {e}")
        return None


OCR_VARIANT_BUILDERS = {
    "clahe": _variant_clahe,
    "clahe_unsharp": _variant_clahe_unsharp,
    "sauvola_strong": _variant_sauvola_strong,
}


# ==============================================================================
# VALIDACIÓN Y NORMALIZACIÓN
# ==============================================================================
def matches_official_format(text):
    """True si el texto cumple alguno de los formatos oficiales (regex estricto)."""
    if not text:
        return False
    return any(p.match(text) for p in PLATE_REGEX_PATTERNS)


def is_valid_plate(text, strict=None):
    """Valida una lectura contra los formatos de patente argentina.

    strict=True (producción, --strict-plate): exige LLLNNN, LLNNNLL o LNNNLLL,
    de modo que una mancha no pueda pasar por letra donde va un número. ES la
    validación de formato argentino.
    strict=False (SOLO debug): longitud 6-7 y alfanumérico en mayúsculas. NO
    equivale a formato argentino válido: "ABCDEFG" o "123456" pasan.
    strict=None usa STRICT_PLATE_VALIDATION. [FIX-v5 #7] El pipeline pasa el
    valor de --strict-plate.
    """
    if not text:
        return False
    if strict is None:
        strict = STRICT_PLATE_VALIDATION
    if strict:
        return matches_official_format(text)
    # [FIX-v3 #7] Antes: len(text) in (6,7) and text.isalnum() and text.isupper().
    # isupper() es False cuando no hay caracteres con caja, así que "123456" se
    # rechazaba por un motivo equivocado (y no por no ser una patente válida).
    return bool(DEBUG_PLATE_REGEX.match(text))


def smart_correct_plate(text, vehicle_class=None):
    """Aplica correcciones OCR solo donde el formato lo justifique.

    [FIX-v5 #8] Con 7 caracteres, LLNNNLL (auto) y LNNNLLL (moto) pueden dar la
    misma penalización. Antes el empate iba siempre a auto; ahora, si el
    vehículo es una moto, el empate se resuelve a favor de LNNNLLL.
    """
    if not text:
        return text

    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    length = len(text)

    # Mapa conservador: solo confusiones visualmente muy parecidas en chapas
    # argentinas, para no introducir correcciones erróneas.
    map_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '4': 'A', '6': 'G'}
    map_to_number = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'A': '4', 'G': '6'}

    def apply_format(raw, mask):
        corrected = []
        changes = 0
        for char, m in zip(raw, mask):
            if m == 'L':
                if char.isalpha():
                    corrected.append(char)
                elif char in map_to_letter:
                    corrected.append(map_to_letter[char])
                    changes += 1
                else:
                    corrected.append(char)
                    changes += 10  # penalización pesada si no se puede arreglar
            else:  # m == 'N'
                if char.isdigit():
                    corrected.append(char)
                elif char in map_to_number:
                    corrected.append(map_to_number[char])
                    changes += 1
                else:
                    corrected.append(char)
                    changes += 10
        return "".join(corrected), changes

    if length == 6:
        corrected, _ = apply_format(text, "LLLNNN")
        return corrected
    if length == 7:
        c1, penalties1 = apply_format(text, "LLNNNLL")
        c2, penalties2 = apply_format(text, "LNNNLLL")
        if penalties1 == penalties2:
            return c2 if vehicle_class == "motorcycle" else c1
        return c1 if penalties1 < penalties2 else c2

    return text


def _fragment_geometry(result):
    """Devuelve (cx, cy, alto) de un fragmento de EasyOCR."""
    xs = [p[0] for p in result[0]]
    ys = [p[1] for p in result[0]]
    return float(np.mean(xs)), float(np.mean(ys)), float(max(ys) - min(ys))


def select_ocr_result(ocr_results):
    """Fusiona los fragmentos de EasyOCR que componen UNA patente.

    Antes se devolvía solo el fragmento de mayor confianza, así que una chapa
    con separación visual ("AB 123 CD") devolvía medio dominio.

    [FIX-v3 #3] La v2 fusionaba únicamente dentro de la misma línea horizontal,
    lo que dejaba rotas las chapas de moto, que son de DOS líneas. Ahora:
      1) Se descartan fragmentos de confianza ínfima.
      2) Se descartan los fragmentos bajos respecto del más alto: así se va el
         texto chico de la banda azul (MERCOSUR / ARGENTINA) y los fantasmas,
         que son la fuente clásica de caracteres de más.
      3) Se agrupan en líneas por posición vertical, se ordenan las líneas de
         arriba hacia abajo y los fragmentos de izquierda a derecha.
      4) Si la concatenación completa supera OCR_MAX_MERGED_LEN, no es una
         patente: se cae a la línea que contiene el fragmento más confiable.

    La confianza devuelta es el promedio ponderado por longitud de texto.
    """
    if not ocr_results:
        return None

    frags = [r for r in ocr_results if r[2] >= OCR_MIN_FRAGMENT_CONF and r[1]]
    if not frags:
        return None
    if len(frags) == 1:
        return frags[0]

    geoms = {id(r): _fragment_geometry(r) for r in frags}
    max_h = max(geoms[id(r)][2] for r in frags)
    if max_h > 0:
        frags = [r for r in frags if geoms[id(r)][2] >= OCR_MIN_HEIGHT_RATIO * max_h]
    if not frags:
        return None

    # Agrupar en líneas por proximidad vertical.
    frags.sort(key=lambda r: geoms[id(r)][1])
    lines = []
    current = [frags[0]]
    for prev, cur in zip(frags, frags[1:]):
        _, prev_cy, prev_h = geoms[id(prev)]
        _, cur_cy, cur_h = geoms[id(cur)]
        ref_h = max(prev_h, cur_h, 1.0)
        if (cur_cy - prev_cy) > OCR_LINE_SPLIT_RATIO * ref_h:
            lines.append(current)
            current = [cur]
        else:
            current.append(cur)
    lines.append(current)

    for line in lines:
        line.sort(key=lambda r: min(p[0] for p in r[0]))

    def build(selected_lines):
        picked = [r for line in selected_lines for r in line]
        text = "".join(r[1] for r in picked)
        weights = [max(len(r[1]), 1) for r in picked]
        conf = float(np.average([r[2] for r in picked], weights=weights))
        anchor = max(picked, key=lambda r: r[2])
        return (anchor[0], text, conf)

    merged = build(lines)
    if len(merged[1]) <= OCR_MAX_MERGED_LEN:
        return merged

    # Demasiado texto para ser una patente: quedarse con la línea del fragmento
    # más confiable en vez de devolver una concatenación sin sentido.
    best_frag = max(frags, key=lambda r: r[2])
    for line in lines:
        # [FIX-v5 #9] Identidad, no igualdad de tuplas con listas de coordenadas.
        if any(r is best_frag for r in line):
            return build([line])
    return best_frag


# ==============================================================================
# PIPELINE MODULAR
# ==============================================================================
class OcrJob:
    """[BATCH-#1] Candidato a detección de patente, producido en FASE A.

    Encapsula TODO lo que _consume_job necesita para replicar el camino de
    _process_plate_ocr de pipelineV3.py, de modo que la única diferencia entre
    el camino secuencial y el batcheado sea DÓNDE se ejecuta el forward de
    RT-DETR, nunca qué lógica se aplica después.
    """
    __slots__ = ("track", "crop", "vehicle_class", "bbox")

    def __init__(self, track, crop, vehicle_class, bbox):
        self.track = track                    # VehicleTrack asociado
        self.crop = crop                      # recorte del VEHÍCULO (vista del frame)
        self.vehicle_class = vehicle_class    # str: "car", "motorcycle", ...
        self.bbox = bbox                      # np.array [x1, y1, x2, y2] para evidencia


class ALPRPipeline:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tracks = {}
        self.json_log = []

        # [FIX-v3 #8] Contadores separados. La v2 sumaba en una sola variable los
        # descartes a nivel vehículo (previos a RT-DETR) y los descartes a nivel
        # chapa (previos al OCR), y dividía por PASADAS de OCR, que con la
        # segunda pasada en gris pueden ser dos por crop. El porcentaje
        # resultante no medía lo que decía medir.
        self.stats = {
            "vehicle_crops_skipped": 0,   # descartados antes de RT-DETR
            "plate_crops_total": 0,       # crops de chapa producidos por RT-DETR
            "plate_crops_skipped": 0,     # descartados por el focus mask
            "plate_crops_to_ocr": 0,      # llegaron al OCR
            "focus_mask_rejections": defaultdict(int),  # [DIAG] motivo -> cantidad
            # [OCR-ADAPT #6] Métricas del OCR adaptativo. Invariante:
            #   ocr_attempts_total == ocr_valid_readings + ocr_rejected_format
            #                         + ocr_rejected_confidence + ocr_no_text
            "ocr_crops_processed": 0,     # crops que entraron al OCR adaptativo
            "ocr_crops_resolved": 0,      # crops con candidato final (= 1 voto)
            "ocr_attempts_total": 0,      # inferencias de EasyOCR (máx 4 por crop)
            "ocr_attempts_by_variant": defaultdict(int),
            "ocr_selected_by_variant": defaultdict(int),  # variante ganadora por crop
            "ocr_valid_readings": 0,      # intentos con formato válido y confianza ok
            "ocr_rejected_format": 0,     # texto leído pero formato inválido
            "ocr_rejected_confidence": 0,  # formato válido pero confianza < mínimo
            "ocr_no_text": 0,             # EasyOCR no devolvió texto (o falló)
        }
        # [BATCH-#2/#4] Lote máximo por forward de RT-DETR + telemetría propia
        # del camino batcheado (para la tesis interesa el throughput real:
        # crops por forward y pico de VRAM, no solo el FPS de pared).
        self.rtdetr_batch_size = max(1, int(getattr(args, "rtdetr_batch", DEFAULT_RTDETR_BATCH)))
        # [FIX-v5 #3] Lote efectivo: baja ante OOM y se RECUERDA entre frames.
        self._rtdetr_batch_effective = self.rtdetr_batch_size
        self.stats.update({
            "rtdetr_forwards": 0,
            "rtdetr_crops_in_batches": 0,
            "rtdetr_oom_backoffs": 0,
            "tracks_flushed": 0,          # [FIX-v5 #4] finalizados por vaciado
            "records_consensus": 0,       # [POLICY] registradas con finalization="consensus"
            "records_recovered": 0,       # [POLICY] registradas con track_lost/end_of_video
            "records_format_invalid": 0,  # [POLICY] registradas con format_valid=False
            "records_rejected_invalid": 0,  # [EXPORT #2] evidencia/patente inconsistente
            "export_audit_violations": 0,   # [EXPORT #5]
        })

        # [FIX-v5 #4/#7] Opciones nuevas, con fallback si args no las trae.
        self.strict_plate = bool(getattr(args, "strict_plate", STRICT_PLATE_VALIDATION))
        self.flush_unfinalized = bool(getattr(args, "flush_unfinalized", FLUSH_UNFINALIZED_DEFAULT))

        # [OCR-ADAPT #5] Presupuesto de intentos: 1 base + hasta 3 variantes.
        # --no-dual-ocr deja solo el intento base.
        max_total = 1 + len(OCR_VARIANT_ORDER)
        self.ocr_max_attempts = min(max_total, max(
            1, int(getattr(args, "ocr_max_attempts", OCR_MAX_ATTEMPTS))))
        if bool(getattr(args, "dual_ocr", True)):
            self._ocr_variants = OCR_VARIANT_ORDER[:self.ocr_max_attempts - 1]
        else:
            self._ocr_variants = ()

        self.recent_plates = {}
        self._jsonl_handle = None

        # [FIX-DIAG] force=True: si alguna libreria importada antes (easyocr,
        # torch, etc.) ya le puso un handler al root logger, basicConfig()
        # sin esto queda como no-op y TODOS nuestros logging.info() del
        # pipeline (incluido el resumen final de metricas) desaparecen sin
        # error ni aviso -- el proceso corre y termina bien, pero en
        # silencio total. force=True (Python >=3.8) descarta cualquier
        # handler previo y garantiza que esta configuracion sea la que rige.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)

        # [BATCH-#4] TF32 en Ampere. [FIX-v5 #6] cudnn.benchmark pasó a opt-in:
        # el autotuner se dispara con cada shape nueva, y EasyOCR (CRAFT +
        # reconocedor) recibe shapes distintas en cada crop, así que puede
        # costar más de lo que ahorra. Compará con --cudnn-benchmark.
        torch.backends.cudnn.benchmark = bool(getattr(args, "cudnn_benchmark", False))
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.reset_peak_memory_stats()

        self._validate_paths()
        self._load_models()
        self._init_trackers_and_annotators()

    # --------------------------------------------------------------------------
    # Inicialización
    # --------------------------------------------------------------------------
    def _validate_paths(self):
        if not os.path.exists(self.args.source):
            logging.error(f"El video de entrada no existe: {self.args.source}")
            raise FileNotFoundError(self.args.source)
        if not os.path.exists(self.args.plate_model):
            logging.error(f"El modelo RT-DETR no existe: {self.args.plate_model}")
            raise FileNotFoundError(self.args.plate_model)

        os.makedirs(self.args.snapshots_dir, exist_ok=True)
        if self.args.debug_crops:
            self.debug_dir = os.path.join(self.args.snapshots_dir, "debug")
            os.makedirs(self.debug_dir, exist_ok=True)
        else:
            self.debug_dir = None

        out_json_dir = os.path.dirname(self.args.output_json)
        if out_json_dir:
            os.makedirs(out_json_dir, exist_ok=True)

        out_vid_dir = os.path.dirname(self.args.output_video)
        if out_vid_dir:
            os.makedirs(out_vid_dir, exist_ok=True)

    def _threshold_values(self):
        g = globals()
        return ", ".join(f"{n}={g[n]}" for n in UNCALIBRATED_THRESHOLDS)

    def _log_policy_banner(self):
        """[POLICY] Deja explícito, al arrancar, el modo de validación y qué
        umbrales siguen sin calibrar. No altera la lógica."""
        if self.strict_plate:
            logging.info("[POLÍTICA] Validación de patente: ESTRICTA "
                         "(LLLNNN / LLNNNLL / LNNNLLL).")
        else:
            logging.warning("[POLÍTICA] Validación de patente: PERMISIVA/DEBUG "
                            "(cualquier cadena A-Z0-9 de 6-7 caracteres). strict=False NO equivale a "
                            "formato argentino válido. Para producción usar --strict-plate.")
        logging.warning("[POLÍTICA] Umbrales SIN CALIBRAR (valores iniciales estimados, no finales): %s",
                        self._threshold_values())
        logging.info("[POLÍTICA] Recuperación sin consenso (track perdido / fin de video): %s. "
                     "Esas detecciones se marcan en 'finalization' y no equivalen a 'consensus'.",
                     "HABILITADA" if self.flush_unfinalized else "deshabilitada")

    def _load_models(self):
        self._log_policy_banner()
        logging.info(f"Utilizando dispositivo de base: {self.device.upper()}")
        # [FIX-v5 #11] Movido desde el nivel de módulo (ahí no se veía).
        logging.info(f"[VEHICLE_CLASS_NAMES] Mapeo real resuelto desde rfdetr: {VEHICLE_CLASS_NAMES}")
        logging.info(f"[OCR-ADAPT] Intento base + variantes habilitadas: "
                     f"{list(self._ocr_variants)} (máx {1 + len(self._ocr_variants)} intentos/crop)")

        logging.info("Cargando modelo RF-DETR (Vehículos)...")
        try:
            self.vehicle_model = RFDETRMedium()
            if self.device == "cuda":
                if hasattr(self.vehicle_model, 'to'):
                    self.vehicle_model.to(self.device)
                    logging.info("RF-DETR movido explícitamente a CUDA mediante .to().")
                else:
                    logging.info("RF-DETR cargado. No expone .to(); el dispositivo real "
                                 "de inferencia depende de su implementación interna.")
            else:
                logging.info("Ejecución en CPU para RF-DETR.")
        except Exception as e:
            logging.error(f"Error crítico al inicializar RF-DETR: {e}", exc_info=True)
            raise

        logging.info("Cargando modelo RT-DETR (Patentes)...")
        self.plate_processor = RTDetrImageProcessor.from_pretrained(self.args.plate_model)
        self.plate_model = RTDetrForObjectDetection.from_pretrained(self.args.plate_model).to(self.device)
        self.plate_model.eval()

        logging.info("Cargando EasyOCR...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"), verbose=False)

    def _init_trackers_and_annotators(self):
        self.video_info = sv.VideoInfo.from_video_path(self.args.source)
        self.roi_y = int(self.video_info.resolution_wh[1] * OCR_ZONE_START_RATIO)

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=30,
            frame_rate=self.video_info.fps
        )
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

    # --------------------------------------------------------------------------
    # Detección
    # --------------------------------------------------------------------------
    def _detect_vehicles(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_veh = self.vehicle_model.predict(frame_rgb, threshold=CONFIDENCE_VEHICLE)

        if hasattr(raw_veh, 'xyxy') and len(raw_veh.xyxy) > 0:
            dets = sv.Detections(
                xyxy=np.array(raw_veh.xyxy),
                confidence=np.array(raw_veh.confidence),
                class_id=np.array(raw_veh.class_id).astype(int),
            )
            dets = dets[np.isin(dets.class_id, list(VEHICLE_CLASS_NAMES.keys()))]
            return dets
        return sv.Detections.empty()

    def _detect_plate_box(self, vehicle_crop):
        """Forward de RT-DETR sobre el recorte del vehículo. Devuelve el box o None.

        [FIX-v3 #4] El forward Y el post-proceso ocurren dentro del mismo bloque
        inference_mode, y lo que sale ya es numpy. En la v2 el post-proceso
        quedaba afuera: los tensores creados bajo inference_mode son "inference
        tensors" y operar con ellos fuera del contexto puede tirar RuntimeError,
        con lo cual el pipeline habría perdido el frame entero por el handler de
        excepciones sin que la causa fuera evidente.
        """
        crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
        inputs = self.plate_processor(images=crop_rgb, return_tensors="pt").to(self.device)
        target_sizes = torch.tensor([vehicle_crop.shape[:2]]).to(self.device)

        with torch.inference_mode():
            if self.args.fp16 and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.plate_model(**inputs)
            else:
                outputs = self.plate_model(**inputs)

            results = self.plate_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_PLATE
            )[0]

            if len(results["scores"]) == 0:
                return None
            scores = results["scores"].float().cpu().numpy()
            boxes = results["boxes"].float().cpu().numpy()

        return boxes[int(np.argmax(scores))]

    # [FIX-v5 #1] Estaba indentado 8 espacios (anidado dentro de
    # _detect_plate_box, después del return): AttributeError o IndentationError.
    def _detect_plate_box_batch(self, vehicle_crops):
        """[BATCH-#1] Forward batcheado de RT-DETR sobre recortes de vehículos.

        Devuelve una lista alineada 1:1 con vehicle_crops: el box de mayor
        confianza por recorte, o None si no hubo detección (misma semántica
        que _detect_plate_box de V3, solo que N a la vez).

        [FIX-BATCH] Las conversiones a numpy ocurren DENTRO del bloque
        inference_mode: los resultados del post-proceso son inference tensors
        y convertirlos fuera del contexto es exactamente el riesgo que
        documentó [FIX-v3 #4] para la versión secuencial.
        """
        if not vehicle_crops:
            return []

        crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in vehicle_crops]
        inputs = self.plate_processor(images=crops_rgb, return_tensors="pt").to(self.device)
        # target_sizes = (h, w) ORIGINALES de cada crop: el post-proceso
        # reescala los boxes normalizados a las coordenadas de cada recorte.
        target_sizes = torch.tensor([c.shape[:2] for c in vehicle_crops]).to(self.device)

        batch_boxes = []
        with torch.inference_mode():
            if self.args.fp16 and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.plate_model(**inputs)
            else:
                outputs = self.plate_model(**inputs)

            results = self.plate_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_PLATE
            )

            for res in results:
                if len(res["scores"]) == 0:
                    batch_boxes.append(None)
                else:
                    scores = res["scores"].float().cpu().numpy()
                    boxes = res["boxes"].float().cpu().numpy()
                    batch_boxes.append(boxes[int(np.argmax(scores))])

        return batch_boxes

    # --------------------------------------------------------------------------
    # OCR  [OCR-ADAPT]
    # --------------------------------------------------------------------------
    def _ocr_pass(self, image):
        """Una inferencia de EasyOCR con allowlist, tolerante a fallos.

        [OCR-ADAPT] El conteo de intentos vive en _ocr_attempt (que sabe a qué
        variante pertenece la inferencia), no acá.
        """
        if image is None or image.size == 0:
            return None
        try:
            results = self.ocr_reader.readtext(
                image, detail=1,
                allowlist=OCR_ALLOWLIST,
                text_threshold=OCR_TEXT_THRESHOLD,
                low_text=OCR_LOW_TEXT,
            )
        except Exception as e:
            logging.debug(f"[ocr] Pasada fallida: {e}")
            return None
        return select_ocr_result(results)

    def _ocr_attempt(self, variant, image, vehicle_class):
        """[OCR-ADAPT #1/#6] UN intento de OCR sobre `image`.

        Aplica smart_correct_plate() + is_valid_plate() a la salida del
        intento y actualiza los contadores. Devuelve un dict candidato si la
        lectura es válida (formato Y confianza), o None. Si la imagen de la
        variante no pudo construirse NO se ejecuta EasyOCR y NO cuenta como
        intento.
        """
        if image is None or image.size == 0:
            logging.debug(f"[ocr_adapt] variante '{variant}' sin imagen: no se ejecuta")
            return None

        s = self.stats
        s["ocr_attempts_total"] += 1
        s["ocr_attempts_by_variant"][variant] += 1

        best = self._ocr_pass(image)
        if best is None:
            s["ocr_no_text"] += 1
            return None

        raw_text, conf = best[1], float(best[2])
        text = smart_correct_plate(raw_text, vehicle_class)

        if not is_valid_plate(text, self.strict_plate):
            s["ocr_rejected_format"] += 1
            logging.debug(f"[ocr_adapt] '{variant}': rechazado por formato "
                          f"raw={raw_text!r} corregido={text!r} conf={conf:.2f}")
            return None
        if conf < OCR_CANDIDATE_MIN_CONF:
            s["ocr_rejected_confidence"] += 1
            logging.debug(f"[ocr_adapt] '{variant}': rechazado por confianza "
                          f"{text!r} conf={conf:.2f} < {OCR_CANDIDATE_MIN_CONF}")
            return None

        s["ocr_valid_readings"] += 1
        logging.debug(f"[ocr_adapt] '{variant}': candidato válido {text!r} conf={conf:.2f}")
        return {
            "text": text,
            "conf": conf,
            "variant": variant,
            "processed": image,                       # lo que realmente vio EasyOCR
            "official": matches_official_format(text),
        }

    def _read_plate_adaptive(self, plate_crop, vehicle_class):
        """[OCR-ADAPT #1-#4] OCR adaptativo sobre UN crop de patente.

        Devuelve (texto_validado, confianza, imagen_procesada) con el ÚNICO
        candidato final, o None si ningún intento produjo una lectura válida
        (el llamador no vota ni toca el track: el crop simplemente se
        descarta y se reintenta en un frame posterior).

        Garantías:
          - Máximo 1 + len(self._ocr_variants) <= OCR_MAX_ATTEMPTS inferencias.
          - Todos los intentos parten del mismo plate_crop natural (el prewarp
            es una corrección geométrica compartida, no un filtro encadenado).
          - Corte temprano apenas hay un candidato válido con confianza
            >= OCR_SUFFICIENT_CONF.
          - Selección: preferencia por formato oficial y luego mayor confianza.
        """
        s = self.stats
        s["ocr_crops_processed"] += 1

        warped = _prewarp_plate(plate_crop)   # base geométrica común
        candidates = []

        # Intento 1: pipeline actual de preprocesamiento.
        cand = self._ocr_attempt(OCR_VARIANT_BASE, _preprocess_plate_for_ocr(warped), vehicle_class)
        if cand is not None:
            candidates.append(cand)

        # Intentos 2..4: variantes independientes, solo mientras no haya un
        # candidato válido con confianza suficiente.
        for variant in self._ocr_variants:
            if any(c["conf"] >= OCR_SUFFICIENT_CONF for c in candidates):
                break
            image = OCR_VARIANT_BUILDERS[variant](warped)
            cand = self._ocr_attempt(variant, image, vehicle_class)
            if cand is not None:
                candidates.append(cand)

        if not candidates:
            return None

        # UN resultado por crop: formato oficial primero, luego confianza.
        winner = max(candidates, key=lambda c: (c["official"], c["conf"]))
        s["ocr_crops_resolved"] += 1
        s["ocr_selected_by_variant"][winner["variant"]] += 1
        return winner["text"], winner["conf"], winner["processed"]

    def _process_plate_ocr(self, vehicle_crop, vehicle_class):
        """Devuelve (texto_validado, confianza, plate_crop_natural, procesado)."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, 0.0, None, None

        # Focus mask temprana sobre el vehículo: si es más chico que esto, no
        # puede contener una chapa legible y nos ahorramos también el forward
        # de RT-DETR, no solo el OCR.
        vh, vw = vehicle_crop.shape[:2]
        if not self.args.disable_focus_mask and (vw < MIN_VEHICLE_CROP_WIDTH or vh < MIN_VEHICLE_CROP_HEIGHT):
            self.stats["vehicle_crops_skipped"] += 1
            logging.debug(
                f"[focus_mask] Vehiculo descartado ANTES de RT-DETR por tamano "
                f"({vw}x{vh} < minimo {MIN_VEHICLE_CROP_WIDTH}x{MIN_VEHICLE_CROP_HEIGHT}). "
                f"Clase={vehicle_class}"
            )
            return None, 0.0, None, None

        box_p = self._detect_plate_box(vehicle_crop)
        if box_p is None:
            return None, 0.0, None, None

        ch, cw = vehicle_crop.shape[:2]
        px1 = max(0, min(int(box_p[0]), cw))
        py1 = max(0, min(int(box_p[1]), ch))
        px2 = max(0, min(int(box_p[2]), cw))
        py2 = max(0, min(int(box_p[3]), ch))
        if px2 <= px1 or py2 <= py1:
            return None, 0.0, None, None

        plate_crop = vehicle_crop[py1:py2, px1:px2]
        self.stats["plate_crops_total"] += 1

        # EVIDENCIA vs ENTRADA DEL OCR:
        #   - plate_crop (natural) es lo que se guarda como evidencia en disco,
        #     porque es lo que un revisor humano espera poder auditar.
        #   - lo que recibe EasyOCR es una COPIA derivada (prewarp + filtros de
        #     cada intento). El natural y el procesado nunca se mezclan.
        if not self.args.disable_focus_mask:
            ok, reason = _has_plate_characteristics(plate_crop, vehicle_class)
            if not ok:
                self.stats["plate_crops_skipped"] += 1
                self.stats["focus_mask_rejections"][reason] += 1
                logging.debug(
                    f"[focus_mask] Patente detectada por RT-DETR pero descartada "
                    f"antes del OCR. Clase={vehicle_class} motivo={reason}"
                )
                return None, 0.0, plate_crop, None

        self.stats["plate_crops_to_ocr"] += 1
        # [OCR-ADAPT] Candidato final único (ya corregido y validado).
        result = self._read_plate_adaptive(plate_crop, vehicle_class)
        if result is None:
            return None, 0.0, plate_crop, None

        text, conf, processed = result
        return text, conf, plate_crop, processed

    # --------------------------------------------------------------------------
    # Finalización y salida  [EXPORT]
    # --------------------------------------------------------------------------
    def _evidence_problem(self, ev, plate):
        """[EXPORT #2] Devuelve un motivo (str) si la evidencia NO es utilizable
        para exportar `plate`, o None si es consistente."""
        if ev is None:
            return "sin evidencia para la patente ganadora"
        if ev.text != plate:
            return f"evidencia de otra patente ({ev.text!r} != {plate!r})"
        if ev.frame is None or ev.frame < 0:
            return f"frame_idx inválido ({ev.frame})"
        if ev.crop is None or ev.crop.size == 0:
            return "sin recorte natural"
        if ev.bbox is None or len(ev.bbox) != 4:
            return "sin bounding box"
        return None

    def _reject_record(self, v_track, plate, why):
        self.stats["records_rejected_invalid"] += 1
        logging.error(f"[export] ID #{v_track.track_id} NO se registra "
                      f"(patente={plate!r}): {why}")

    def _finalize_track(self, v_track, reason="consensus"):
        """reason: "consensus" (criterios normales), "track_lost" o
        "end_of_video" (vaciado, ver _flush_track). Queda en el registro.

        [EXPORT #4] Devuelve True solo si se agregó un registro a json_log
        (False si fue duplicado o fue rechazado por inconsistencia).
        """
        v_track.state = TrackState.FINALIZED
        best_plate, avg_conf, count = v_track.get_consensus_plate()

        v_track.best_plate_text = best_plate

        # [EXPORT #3] Solo se exporta una patente que pase la validación configurada.
        if not best_plate or not is_valid_plate(best_plate, self.strict_plate):
            self._reject_record(v_track, best_plate, "no pasa la validación de patente configurada")
            return False

        # [EXPORT #2] La evidencia debe ser la de la patente ganadora y estar completa.
        # .get(): evidence es un defaultdict y [] crearía una entrada vacía.
        ev = v_track.evidence.get(best_plate)
        problem = self._evidence_problem(ev, best_plate)
        if problem is None and not (self.video_info.fps and self.video_info.fps > 0):
            problem = f"fps inválido ({self.video_info.fps})"
        if problem is not None:
            self._reject_record(v_track, best_plate, problem)
            return False

        timestamp = ev.frame / float(self.video_info.fps)   # >= 0 por construcción

        # Deduplicación de tracks re-identificados: si ByteTrack pierde y
        # recupera el mismo vehículo aparecen dos IDs con la misma patente.
        if self.args.dedup_seconds > 0:
            last_seen = self.recent_plates.get(best_plate)
            if last_seen is not None and (timestamp - last_seen) <= self.args.dedup_seconds:
                logging.info(f"[DUPLICADO] ID #{v_track.track_id} -> {best_plate} "
                             f"(visto hace {timestamp - last_seen:.1f}s, no se registra)")
                self.recent_plates[best_plate] = timestamp
                return False
        self.recent_plates[best_plate] = timestamp

        # El snapshot se genera desde el recorte NATURAL de la MISMA evidencia
        # que aporta frame_idx, timestamp y bounding_box: la binarizada solo la
        # vio el OCR y nunca se guarda como evidencia.
        snap_filename = f"{best_plate}_ID{v_track.track_id}_F{ev.frame}.jpg"
        snap_path = os.path.join(self.args.snapshots_dir, snap_filename)
        if not cv2.imwrite(snap_path, ev.crop):
            logging.warning(f"[export] No se pudo escribir el snapshot: {snap_path}")
            snap_path = None

        # [MEJORA-v3 #11] Volcado de diagnóstico: natural vs. lo que vio el
        # OCR. Sirve para las figuras de la tesis y para entender una mala
        # lectura sin tener que reproducir la corrida entera.
        if snap_path and self.debug_dir and ev.processed is not None and ev.processed.size > 0:
            dbg_name = f"{best_plate}_ID{v_track.track_id}_F{ev.frame}_ocr.png"
            cv2.imwrite(os.path.join(self.debug_dir, dbg_name), ev.processed)

        record = {
            "id_vehiculo": int(v_track.track_id),            # track_id de ByteTrack
            "tipo_vehiculo": v_track.vehicle_class_name,     # clase de RF-DETR
            "patente_validada": best_plate,
            "confianza_ocr_promedio": round(float(avg_conf), 3),
            "cantidad_votos": int(count),
            "timestamp_seconds": round(float(timestamp), 2),
            "frame_idx": int(ev.frame),
            "snapshot_path": snap_path,
            "bounding_box": [int(x) for x in ev.bbox],
            # [FIX-v5 #4/#7] Trazabilidad de calidad para filtrar en el análisis.
            "finalization": reason,
            "format_valid": bool(matches_official_format(best_plate)),
        }
        self.json_log.append(record)
        self._append_jsonl(record)

        # [POLICY] El log y los contadores separan consenso de recuperada.
        if reason == CONSENSUS_REASON:
            self.stats["records_consensus"] += 1
            tag = "NUEVO ALPR"
        else:
            self.stats["records_recovered"] += 1
            tag = "RECUPERADO SIN CONSENSO"
        if not record["format_valid"]:
            self.stats["records_format_invalid"] += 1
        logging.info(f"[{tag}] ID #{v_track.track_id} ({v_track.vehicle_class_name}) -> "
                     f"{best_plate} (Conf: {avg_conf:.2f}, Votos: {count}, {reason}, "
                     f"format_valid={record['format_valid']})")
        return True

    def _append_jsonl(self, record):
        """Persistencia incremental: si el proceso muere, no se pierde todo."""
        if self._jsonl_handle is None:
            return
        try:
            self._jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_handle.flush()
        except Exception as e:
            logging.warning(f"[jsonl] No se pudo volcar la detección incremental: {e}")

    def _audit_detections(self):
        """[EXPORT #5] Verifica TODAS las detecciones antes de escribir el JSON.

        Es una segunda barrera: _finalize_track ya rechaza lo inconsistente, así
        que en una corrida sana devuelve []. Comprueba campos obligatorios,
        timestamp_seconds >= 0, frame_idx >= 0, bounding_box de 4 enteros,
        patente válida y que snapshot_path corresponda a la patente y al frame
        del registro (misma evidencia). Devuelve la lista de violaciones.
        """
        problems = []
        for rec in self.json_log:
            tag = f"ID #{rec.get('id_vehiculo')} ({rec.get('patente_validada')})"
            missing = [k for k in EXPORT_FIELDS if k not in rec]
            if missing:
                problems.append(f"{tag}: faltan campos {missing}")
                continue
            if not isinstance(rec["timestamp_seconds"], (int, float)) or rec["timestamp_seconds"] < 0:
                problems.append(f"{tag}: timestamp_seconds inválido ({rec['timestamp_seconds']})")
            if not isinstance(rec["frame_idx"], int) or rec["frame_idx"] < 0:
                problems.append(f"{tag}: frame_idx inválido ({rec['frame_idx']})")
            bbox = rec["bounding_box"]
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)):
                problems.append(f"{tag}: bounding_box inválido ({bbox})")
            if not is_valid_plate(rec["patente_validada"], self.strict_plate):
                problems.append(f"{tag}: patente no pasa la validación configurada")
            if rec["finalization"] not in FINALIZATION_REASONS:
                problems.append(f"{tag}: finalization desconocido ({rec['finalization']})")
            snap = rec["snapshot_path"]
            if snap is not None:
                name = os.path.basename(snap)
                if not name.startswith(f"{rec['patente_validada']}_ID{rec['id_vehiculo']}_F"):
                    problems.append(f"{tag}: snapshot de otra patente/track ({name})")
                if not name.endswith(f"_F{rec['frame_idx']}.jpg"):
                    problems.append(f"{tag}: snapshot de otro frame ({name})")
        for p in problems:
            logging.error(f"[export][AUDITORÍA] {p}")
        self.stats["export_audit_violations"] = len(problems)
        if not problems:
            logging.info(f"[export][AUDITORÍA] {len(self.json_log)} detecciones verificadas sin violaciones")
        return problems

    def _draw_annotations(self, frame, tracked):
        labels = []
        # [FIX-v5 #11] tracker_id puede ser None si no hay detecciones.
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for t_id in tracker_ids:
            if t_id is not None and t_id in self.tracks:
                v_track = self.tracks[t_id]
                if v_track.state == TrackState.FINALIZED:
                    display_text = v_track.best_plate_text
                else:
                    plate, _, _ = v_track.get_consensus_plate()
                    display_text = plate if plate else v_track.vehicle_class_name
                labels.append(f"#{t_id} | {display_text}")
            else:
                labels.append("...")

        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=tracked)
        annotated = self.label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

        cv2.line(annotated, (0, self.roi_y),
                 (self.video_info.resolution_wh[0], self.roi_y), (0, 0, 255), 2)
        cv2.putText(annotated, "ZONA LECTURA OCR", (10, self.roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated

    def _purge_stale_tracks(self, current_frame, active_ids):
        stale_ids = [
            tid for tid, t in self.tracks.items()
            if (current_frame - t.last_seen_frame) > TRACK_TTL_FRAMES and tid not in active_ids
        ]
        for tid in stale_ids:
            # [FIX-v5 #4] Antes se borraba sin más: una patente leída sin
            # alcanzar consenso desaparecía sin dejar rastro.
            self._flush_track(self.tracks[tid], "track_lost")
            del self.tracks[tid]

    def _flush_track(self, v_track, reason):
        """[FIX-v5 #4] Finaliza un track que se cierra SIN haber alcanzado consenso.

        Criterio deliberadamente más conservador que el normal para no meter
        basura en los datos:
          - Se necesita alguna lectura con confianza media >= MIN_AVG_OCR_CONFIDENCE.
          - Con >= MIN_VOTES_TO_FINALIZE votos se acepta el ganador aunque el
            margen no llegue (el registro dice "finalization": reason).
          - Con UNA sola lectura, se acepta solo si cumple el formato oficial
            (regex estricto) y supera FLUSH_SINGLE_VOTE_MIN_CONF.
        [POLICY] El resultado es una detección RECUPERADA, no confirmada: queda
        con finalization="track_lost"/"end_of_video" y su calidad no es
        equivalente a la de "consensus". Los umbrales usados acá están [SIN CALIBRAR].
        [EXPORT #4] Devuelve True solo si se registró de verdad una detección.
        """
        if not self.flush_unfinalized or v_track.state != TrackState.OCR_ACTIVE:
            return False
        plate, avg_conf, count = v_track.get_consensus_plate()
        if not plate or avg_conf < MIN_AVG_OCR_CONFIDENCE:
            return False
        if count < MIN_VOTES_TO_FINALIZE:
            if not (matches_official_format(plate) and avg_conf >= FLUSH_SINGLE_VOTE_MIN_CONF):
                return False
        recorded = self._finalize_track(v_track, reason=reason)
        if recorded:
            self.stats["tracks_flushed"] += 1
        return recorded

    def _flush_all_tracks(self, reason):
        for v_track in list(self.tracks.values()):
            self._flush_track(v_track, reason)

    # --------------------------------------------------------------------------
    # Loop principal
    # --------------------------------------------------------------------------
    def process_video(self):
        """[BATCH-#1] Por frame: (1) RF-DETR + ByteTrack como V3,
        (2) FASE A, (3) FASE B, (4) purga + anotación + escritura."""
        frame_idx = 0
        t_start = time.time()
        gen = sv.get_video_frames_generator(source_path=self.args.source)

        sink = (sv.VideoSink(target_path=self.args.output_video, video_info=self.video_info)
                if self.args.output_video else None)
        if sink is not None:
            sink.__enter__()
        try:
            for frame in gen:
                if self.args.max_frames and frame_idx >= self.args.max_frames:
                    break

                detections = self._detect_vehicles(frame)
                tracked = self.tracker.update_with_detections(detections)

                jobs = self._collect_ocr_jobs(frame, tracked, frame_idx)   # FASE A
                self._run_ocr_jobs(jobs, frame_idx)                        # FASE B

                tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
                active_ids = {int(t) for t in tracker_ids if t is not None}
                self._purge_stale_tracks(frame_idx, active_ids)

                if sink is not None:
                    sink.write_frame(self._draw_annotations(frame, tracked))

                frame_idx += 1
                if frame_idx % 100 == 0:
                    elapsed = max(time.time() - t_start, 1e-6)
                    logging.info(f"[progreso] {frame_idx} frames | "
                                 f"{frame_idx / elapsed:.2f} FPS prom | "
                                 f"tracks activos={len(self.tracks)} | "
                                 f"jobs ultimo frame={len(jobs)}")
        finally:
            if sink is not None:
                sink.__exit__(None, None, None)

        logging.info(f"[fin] Frames procesados: {frame_idx} "
                     f"en {time.time() - t_start:.1f}s")

    def run(self):
        # Registro incremental: misma política de V3 (flush por registro), un
        # archivo .jsonl hermano del JSON final.
        jsonl_path = os.path.splitext(self.args.output_json)[0] + ".jsonl"
        logging.info(f"[salida] Registro incremental (JSONL): {jsonl_path}")
        self._jsonl_handle = open(jsonl_path, "w", encoding="utf-8")
        try:
            self.process_video()
            # [FIX-v5 #4] Los tracks que siguen abiertos al terminar el video
            # también se vacían (con el handle del JSONL todavía abierto).
            self._flush_all_tracks("end_of_video")
        finally:
            if self._jsonl_handle is not None:
                self._jsonl_handle.close()
                self._jsonl_handle = None

        # [EXPORT #5] Verificación final antes de escribir el JSON.
        self._audit_detections()

        # [EXPORT #1] El JSON final es una LISTA de objetos de detección.
        with open(self.args.output_json, "w", encoding="utf-8") as f:
            json.dump(self.json_log, f, ensure_ascii=False, indent=2)
        logging.info(f"[salida] {len(self.json_log)} detecciones -> {self.args.output_json}")
        self._log_stats()

    def _log_stats(self):
        """[FIX-v3 #8 + BATCH-#4] Métricas: ahorro del focus mask sobre crops
        de CHAPA, descartes de vehículo aparte, y telemetría del batch.

        [FIX-v5 #2] Única definición (antes había dos).

        [DIAG] El desglose por motivo sirve para saber, sin adivinar, cuál de los
        tres checks del focus mask se está comiendo las patentes en un video en
        particular. Si "tamano" domina, hay vehículos/chapas más chicos de lo
        que PLATE_GEOMETRY espera (cámara más lejos). Si domina "aspecto", la
        geometría calibrada no se ajusta a esas placas (ángulo, clase de
        vehículo no prevista). Si domina "desvio_gris", son placas borrosas o
        con poco contraste.

        [OCR-ADAPT #6] Bloque propio con las métricas del OCR adaptativo.
        """
        s = self.stats
        total = s["plate_crops_total"]
        ahorro = (100.0 * s["plate_crops_skipped"] / total) if total else 0.0
        logging.info(
            "[MÉTRICAS] Crops de chapa: %d | descartados por focus mask: %d (%.1f%%) | "
            "enviados a OCR: %d", total, s["plate_crops_skipped"], ahorro,
            s["plate_crops_to_ocr"])
        logging.info("[MÉTRICAS] Vehículos descartados ANTES de RT-DETR: %d",
                     s["vehicle_crops_skipped"])

        crops = s["ocr_crops_processed"]
        if crops:
            order = (OCR_VARIANT_BASE,) + OCR_VARIANT_ORDER
            por_variante = ", ".join(f"{v}={s['ocr_attempts_by_variant'].get(v, 0)}" for v in order)
            ganadoras = ", ".join(f"{v}={s['ocr_selected_by_variant'].get(v, 0)}" for v in order)
            logging.info("[MÉTRICAS][OCR] Crops procesados: %d | con candidato final (1 voto): %d (%.1f%%) | "
                         "sin lectura válida: %d",
                         crops, s["ocr_crops_resolved"], 100.0 * s["ocr_crops_resolved"] / crops,
                         crops - s["ocr_crops_resolved"])
            logging.info("[MÉTRICAS][OCR] Intentos totales: %d (%.2f por crop, máx %d) | por variante: %s",
                         s["ocr_attempts_total"], s["ocr_attempts_total"] / crops,
                         1 + len(self._ocr_variants), por_variante)
            logging.info("[MÉTRICAS][OCR] Lecturas válidas: %d | rechazadas por formato: %d | "
                         "rechazadas por confianza: %d | sin texto: %d",
                         s["ocr_valid_readings"], s["ocr_rejected_format"],
                         s["ocr_rejected_confidence"], s["ocr_no_text"])
            logging.info("[MÉTRICAS][OCR] Variante ganadora por crop: %s", ganadoras)

        logging.info("[MÉTRICAS] Detecciones: %d | por vaciado (track perdido / fin de video): %d | "
                     "rechazadas por evidencia inconsistente: %d | violaciones de auditoría: %d",
                     len(self.json_log), s["tracks_flushed"],
                     s["records_rejected_invalid"], s["export_audit_violations"])
        logging.info("[MÉTRICAS][POLÍTICA] Confirmadas por consenso: %d | RECUPERADAS sin consenso "
                     "(calidad menor, filtrar por 'finalization'): %d | con format_valid=False: %d | "
                     "validación: %s",
                     s["records_consensus"], s["records_recovered"], s["records_format_invalid"],
                     "ESTRICTA" if self.strict_plate else "PERMISIVA/DEBUG (no equivale a formato argentino)")
        logging.info("[MÉTRICAS][POLÍTICA] Umbrales SIN CALIBRAR usados en esta corrida: %s",
                     self._threshold_values())
        if s["rtdetr_forwards"]:
            logging.info(
                "[MÉTRICAS][BATCH] Forwards RT-DETR: %d | crops en lote: %d | "
                "crops/forward medio: %.2f | backoffs por OOM: %d | lote efectivo final: %d",
                s["rtdetr_forwards"], s["rtdetr_crops_in_batches"],
                s["rtdetr_crops_in_batches"] / s["rtdetr_forwards"],
                s["rtdetr_oom_backoffs"], self._rtdetr_batch_effective)
        if s["focus_mask_rejections"]:
            resumen = ", ".join(
                f"{k}={v}" for k, v in
                sorted(s["focus_mask_rejections"].items(), key=lambda kv: -kv[1]))
            logging.info("[MÉTRICAS] Rechazos de focus mask por motivo: %s", resumen)
        if self.device == "cuda":
            logging.info("[MÉTRICAS][VRAM] Pico del proceso: %.2f GB",
                         torch.cuda.max_memory_allocated() / 1024**3)

    # --------------------------------------------------------------------------
    # [BATCH-#1] FASE A: máquina de estados y recolección de jobs (CPU)
    # --------------------------------------------------------------------------
    def _get_or_create_track(self, track_id, vehicle_class_name):
        if track_id not in self.tracks:
            self.tracks[track_id] = VehicleTrack(track_id, vehicle_class_name)
        return self.tracks[track_id]

    def _maybe_activate_ocr(self, v_track, bbox_int):
        """NEW -> OCR_ACTIVE cuando el centro del bbox cruza roi_y (una vez).

        NOTA DE RECONSTRUCCIÓN: el loop de V3 no está en el extracto; asumí
        cruce por CENTRO (cy >= roi_y). Si tu V3 usaba el borde inferior
        (y2 >= roi_y) o detección de cruce direccional, cambiá acá — es el
        único punto del parche que puede no ser 1:1 con V3.
        """
        x1, y1, x2, y2 = bbox_int
        cy = 0.5 * (y1 + y2)
        if cy >= self.roi_y:
            v_track.state = TrackState.OCR_ACTIVE
            logging.debug(f"[estado] Track #{v_track.track_id} "
                          f"({v_track.vehicle_class_name}) entra en zona OCR: "
                          f"cy={cy:.0f} >= roi_y={self.roi_y}")

    def _try_finalize(self, v_track):
        """Finaliza si el consenso ya cumple los criterios probados de V3."""
        if v_track.state != TrackState.OCR_ACTIVE:
            return False
        if v_track.check_finalization():
            self._finalize_track(v_track)
            return True
        return False

    def _collect_ocr_jobs(self, frame, tracked, frame_idx):
        """[BATCH-#1] FASE A: toda la lógica barata SIN tocar RT-DETR.

        Replica la puerta de entrada de _process_plate_ocr de V3 (estado
        OCR_ACTIVE + intervalo + focus mask temprana de vehículo) con las
        MISMAS decisiones y contadores; en vez de ejecutar el forward por
        vehículo, encola un OcrJob. Solo cambia CUÁNDO se ejecuta el forward.
        """
        jobs = []
        if tracked is None or len(tracked) == 0 or tracked.tracker_id is None:
            return jobs

        H, W = frame.shape[:2]
        for bbox, class_id, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
            if track_id is None:
                continue
            track_id = int(track_id)
            v_track = self._get_or_create_track(
                track_id, VEHICLE_CLASS_NAMES.get(int(class_id), "vehicle"))
            v_track.last_seen_frame = frame_idx

            x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
            x2 = min(W, int(bbox[2])); y2 = min(H, int(bbox[3]))
            if x2 <= x1 or y2 <= y1:
                continue

            if v_track.state == TrackState.NEW:
                self._maybe_activate_ocr(v_track, (x1, y1, x2, y2))

            if v_track.state != TrackState.OCR_ACTIVE:
                continue  # NEW sin cruzar la línea, o FINALIZED

            # Consenso alcanzado: finalize inmediato (chequeo barato por frame).
            if self._try_finalize(v_track):
                continue

            if (frame_idx - v_track.last_ocr_frame) < OCR_INTERVAL_FRAMES:
                continue

            vehicle_crop = frame[y1:y2, x1:x2]

            # Focus mask temprana de VEHÍCULO (idéntica a V3): ahorra el
            # forward de RT-DETR además del OCR. Mismo contador de FIX-v3 #8.
            vh, vw = vehicle_crop.shape[:2]
            if not self.args.disable_focus_mask and (
                    vw < MIN_VEHICLE_CROP_WIDTH or vh < MIN_VEHICLE_CROP_HEIGHT):
                self.stats["vehicle_crops_skipped"] += 1
                v_track.last_ocr_frame = frame_idx  # no reintentar cada frame
                logging.debug(f"[focus_mask] Vehiculo descartado ANTES de RT-DETR "
                              f"({vw}x{vh} < minimo {MIN_VEHICLE_CROP_WIDTH}x"
                              f"{MIN_VEHICLE_CROP_HEIGHT}). Clase={v_track.vehicle_class_name}")
                continue

            jobs.append(OcrJob(track=v_track, crop=vehicle_crop,
                               vehicle_class=v_track.vehicle_class_name,
                               bbox=np.array([x1, y1, x2, y2], dtype=np.int32)))
            # Semántica del intervalo: corre desde el INTENTO (encolado), no
            # desde el voto. Si un frame no detecta chapa, no se degrada el
            # intervalo a "un intento por frame".
            v_track.last_ocr_frame = frame_idx

        return jobs

    # --------------------------------------------------------------------------
    # [BATCH-#1/#2] FASE B: forwards batcheados + OCR por job
    # --------------------------------------------------------------------------
    def _detect_plate_box_batch_safe(self, crops):
        """[BATCH-#2] Chunking con backoff ante CUDA OOM.

        En la RTX 3080 el pico de VRAM depende de cuántos modelos residency
        tienen a la vez (RF-DETR + RT-DETR + EasyOCR). Si un lote no entra,
        se vacía caché, se reduce el lote a la mitad y se reintenta el MISMO
        subconjunto: el frame nunca se pierde por memoria.

        [FIX-v5 #3] (a) Solo se captura OOM: cualquier otro RuntimeError se
        re-lanza (antes se seguía con `boxes` sin definir o del chunk previo).
        (b) El lote reducido se guarda en self._rtdetr_batch_effective y se
        respeta en los frames siguientes. (c) Es el ÚNICO lugar donde se
        parte en chunks (antes _run_ocr_jobs también lo hacía).
        """
        n = len(crops)
        results = [None] * n
        i = 0
        size = min(self._rtdetr_batch_effective, n)
        while i < n:
            chunk = crops[i:i + size]
            oom = False
            boxes = None
            try:
                boxes = self._detect_plate_box_batch(chunk)
            except RuntimeError as e:
                # torch.cuda.OutOfMemoryError hereda de RuntimeError.
                is_oom = (isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ()))
                          or "out of memory" in str(e).lower())
                if not is_oom:
                    raise
                oom = True

            if oom:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if size <= 1:
                    logging.error("[batch] OOM incluso con batch=1: "
                                  "se descartan los crops restantes de este frame.")
                    break
                size = max(1, size // 2)
                self._rtdetr_batch_effective = min(self._rtdetr_batch_effective, size)
                self.stats["rtdetr_oom_backoffs"] += 1
                logging.warning(f"[batch] OOM de VRAM en RT-DETR: reintento con batch={size}")
                continue

            self.stats["rtdetr_forwards"] += 1
            self.stats["rtdetr_crops_in_batches"] += len(chunk)
            for k, b in enumerate(boxes):
                results[i + k] = b
            i += len(chunk)
        return results

    def _consume_job(self, job, box_p, frame_idx):
        """[BATCH-#1] Camino de V3 DESPUÉS del forward: clamp del box, recorte
        de chapa, focus mask de chapa, OCR adaptativo y voto.

        [OCR-ADAPT #4] El OCR devuelve UN candidato final ya corregido y
        validado, así que se registra como máximo UN voto por crop/frame. Si
        no hay candidato válido no se vota ni se toca el track: el crop se
        descarta y se reintenta en un frame posterior."""
        if box_p is None:
            return

        vehicle_crop = job.crop
        ch, cw = vehicle_crop.shape[:2]
        px1 = max(0, min(int(box_p[0]), cw))
        py1 = max(0, min(int(box_p[1]), ch))
        px2 = max(0, min(int(box_p[2]), cw))
        py2 = max(0, min(int(box_p[3]), ch))
        if px2 <= px1 or py2 <= py1:
            return

        plate_crop = vehicle_crop[py1:py2, px1:px2]
        self.stats["plate_crops_total"] += 1

        if not self.args.disable_focus_mask:
            ok, reason = _has_plate_characteristics(plate_crop, job.vehicle_class)
            if not ok:
                self.stats["plate_crops_skipped"] += 1
                self.stats["focus_mask_rejections"][reason] += 1
                logging.debug(f"[focus_mask] Patente detectada por RT-DETR pero "
                              f"descartada antes del OCR. Clase={job.vehicle_class} "
                              f"motivo={reason}")
                return

        self.stats["plate_crops_to_ocr"] += 1
        result = self._read_plate_adaptive(plate_crop, job.vehicle_class)
        if result is None:
            return

        text, conf, processed = result
        # PlateEvidence queda sincronizada con el candidato elegido: plate_crop
        # natural + la imagen procesada del intento ganador + frame y bbox.
        job.track.add_vote(text, float(conf), frame_idx,
                           plate_crop, processed, job.bbox)
        self._try_finalize(job.track)

    def _run_ocr_jobs(self, jobs, frame_idx):
        """[BATCH-#1/#2] FASE B: consume los jobs del frame.

        Batcheado: un solo llamado a _detect_plate_box_batch_safe (que parte en
        chunks de tamaño efectivo y hace backoff ante OOM).
        --sequential-rtdetr: línea base V3 (un forward por vehículo) vía
        _process_plate_ocr original, para benchmark A/B honesto.

        [FIX-v5 #1] Se eliminó el `finally:` sin `try:` (con marcas "[cite: 1]")
        que había quedado al final de este método: era un resto de run().
        """
        if not jobs:
            return

        if self.args.sequential_rtdetr:
            for job in jobs:
                texto, conf, plate_crop, processed = self._process_plate_ocr(
                    job.crop, job.vehicle_class)
                if texto:
                    job.track.add_vote(texto, conf, frame_idx,
                                       plate_crop, processed, job.bbox)
                    self._try_finalize(job.track)
            return

        boxes = self._detect_plate_box_batch_safe([job.crop for job in jobs])
        for job, box_p in zip(jobs, boxes):
            self._consume_job(job, box_p, frame_idx)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="ALPR batcheado Fases A/B — UTN-GIBD & Jefatura Dptal. de Policía (RN14)")
    p.add_argument("--source", default=os.path.join("inputs", "trafico.mp4"))
    p.add_argument("--plate-model", default=os.path.join("outputs", "rtdetr-ccpd", "final_model"))
    p.add_argument("--snapshots-dir", default="snapshots")
    p.add_argument("--output-json", default="detecciones_alpr.json")
    p.add_argument("--output-video", default=os.path.join("outputs", "alpr_batch_dev.mp4"))
    p.add_argument("--max-frames", type=int, default=0, help="0 = video completo")
    p.add_argument("--dedup-seconds", type=float, default=30.0)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    # [OCR-ADAPT #5] Antes activaba la segunda pasada en gris. Ahora habilita
    # las variantes de rescate (CLAHE, CLAHE+Unsharp, Sauvola reforzada).
    p.add_argument("--dual-ocr", action=argparse.BooleanOptionalAction, default=True,
                   help="Habilita las variantes de rescate del OCR adaptativo; "
                        "--no-dual-ocr deja solo el intento base")
    p.add_argument("--ocr-max-attempts", type=int, default=OCR_MAX_ATTEMPTS,
                   help="Tope de intentos de EasyOCR por crop de patente (1..4; "
                        "1 = solo intento base). Útil para benchmark A/B")
    p.add_argument("--disable-focus-mask", action="store_true")
    p.add_argument("--debug-crops", action="store_true")
    p.add_argument("--rtdetr-batch", type=int, default=DEFAULT_RTDETR_BATCH,
                   help="Crops de vehículo por forward de RT-DETR (backoff automático ante OOM)")
    p.add_argument("--sequential-rtdetr", action="store_true",
                   help="Línea base V3: un forward por vehículo (benchmark A/B)")
    # [FIX-v5] Opciones nuevas
    p.add_argument("--strict-plate", action=argparse.BooleanOptionalAction,
                   default=STRICT_PLATE_VALIDATION,
                   help="Validación ESTRICTA: solo los formatos oficiales "
                        "(LLLNNN / LLNNNLL / LNNNLLL). Usar en producción. Sin este flag la "
                        "validación es PERMISIVA/DEBUG y NO equivale a formato argentino válido")
    p.add_argument("--flush-unfinalized", action=argparse.BooleanOptionalAction,
                   default=FLUSH_UNFINALIZED_DEFAULT,
                   help="Registra (marcadas en 'finalization') las patentes de tracks "
                        "que se pierden o terminan el video sin alcanzar consenso")
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="Activa torch.backends.cudnn.benchmark (compará A/B: con EasyOCR "
                        "las shapes varían y puede empeorar)")
    # [POLICY] Umbrales de aceptación/finalización sobreescribibles (None = valor
    # del módulo). Todos están [SIN CALIBRAR]; ver UNCALIBRATED_THRESHOLDS.
    p.add_argument("--confidence-vehicle", type=float, default=None,
                   help=f"Umbral RF-DETR vehículos (default {CONFIDENCE_VEHICLE}, sin calibrar)")
    p.add_argument("--confidence-plate", type=float, default=None,
                   help=f"Umbral RT-DETR patentes (default {CONFIDENCE_PLATE}, sin calibrar)")
    p.add_argument("--min-votes", type=int, default=None,
                   help=f"Votos mínimos para finalizar (default {MIN_VOTES_TO_FINALIZE}, sin calibrar)")
    p.add_argument("--min-avg-ocr-conf", type=float, default=None,
                   help=f"Confianza OCR media mínima; también el piso por candidato "
                        f"(default {MIN_AVG_OCR_CONFIDENCE}, sin calibrar)")
    p.add_argument("--early-finalize-votes", type=int, default=None,
                   help=f"Votos de la vía rápida (default {EARLY_FINALIZE_VOTES}, sin calibrar)")
    p.add_argument("--early-finalize-conf", type=float, default=None,
                   help=f"Confianza de la vía rápida (default {EARLY_FINALIZE_CONFIDENCE}, sin calibrar)")
    p.add_argument("--min-consensus-margin", type=float, default=None,
                   help=f"Margen de consenso mínimo (default {MIN_CONSENSUS_MARGIN}, sin calibrar)")
    p.add_argument("--flush-single-vote-conf", type=float, default=None,
                   help=f"Confianza mínima de una lectura única al recuperar un track "
                        f"(default {FLUSH_SINGLE_VOTE_MIN_CONF}, sin calibrar)")
    return p.parse_args()


def _apply_threshold_overrides(args):
    """[POLICY] Aplica los overrides de CLI a las constantes de módulo ANTES de
    crear el pipeline. Sin flags no cambia nada. OCR_CANDIDATE_MIN_CONF sigue
    acoplado a MIN_AVG_OCR_CONFIDENCE (como en el módulo)."""
    global CONFIDENCE_VEHICLE, CONFIDENCE_PLATE, MIN_VOTES_TO_FINALIZE
    global MIN_AVG_OCR_CONFIDENCE, OCR_CANDIDATE_MIN_CONF
    global EARLY_FINALIZE_VOTES, EARLY_FINALIZE_CONFIDENCE
    global MIN_CONSENSUS_MARGIN, FLUSH_SINGLE_VOTE_MIN_CONF
    if args.confidence_vehicle is not None:
        CONFIDENCE_VEHICLE = args.confidence_vehicle
    if args.confidence_plate is not None:
        CONFIDENCE_PLATE = args.confidence_plate
    if args.min_votes is not None:
        MIN_VOTES_TO_FINALIZE = args.min_votes
    if args.min_avg_ocr_conf is not None:
        MIN_AVG_OCR_CONFIDENCE = args.min_avg_ocr_conf
        OCR_CANDIDATE_MIN_CONF = args.min_avg_ocr_conf
    if args.early_finalize_votes is not None:
        EARLY_FINALIZE_VOTES = args.early_finalize_votes
    if args.early_finalize_conf is not None:
        EARLY_FINALIZE_CONFIDENCE = args.early_finalize_conf
    if args.min_consensus_margin is not None:
        MIN_CONSENSUS_MARGIN = args.min_consensus_margin
    if args.flush_single_vote_conf is not None:
        FLUSH_SINGLE_VOTE_MIN_CONF = args.flush_single_vote_conf


def main():
    args = parse_args()
    _apply_threshold_overrides(args)
    pipeline = ALPRPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
