#!/usr/bin/env python3
"""
Clasificador offline de vehículos (Consumidor Desacoplado).

[VERSIÓN CON MEJORAS] Copia de legacy/pipeclasificadorVehiculos.py con las correcciones del reporte
de avance del 15/10/2026 y SIN Ultralytics (AGPL-3.0; ver docs/LICENCIAS.md) (ver standalone/README.md y standalone/pipeclasificadorVehiculos.patch).
El original sigue intacto en legacy/. Equivale a alpr/classify/ pero en un solo archivo.

Arquitectura Productor - Consumidor:

pipelineV5.py (Productor - Fase 4 futura)
      |
      v  (Escritura atómica .tmp -> vehiculo_<track_id>.jpg)
outputs/crops_vehiculos/
      |
      v
 watchdog / PollingObserver (Consumidor - Fase 1)
      |
      v
    Queue
      |
      v
  worker de clasificación (Fase 1: idempotente y consistente)
      |
      +----> clasificador Transformers (Apache-2.0; pesos reales o modo --mock)
      |
      +----> procesados/  (Movimiento físico primero)
      |
      +----> clasificaciones_vehiculos.jsonl (Persistencia fsync)
      |
      +----> fallidos/    (En caso de fallo irrecuperable)

REGLAS DE DISEÑO (Fase 1):
- No modifica pipelineV5.py ni los pipelines históricos.
- No requiere weights/vehicle_classifier si se ejecuta con --mock.
- Si watchdog no está instalado, opera en modo degradado elegante mediante PollingObserver.
- La ayuda (--help) no se interrumpe por librerías opcionales ausentes.
- Manejo transaccional: el archivo se mueve antes de confirmar en JSONL.
- Compatible con etiquetas compuestas mediante catálogo canónico de marcas.
- Protección ante errores de compartición y bloqueo en Windows (WinError 32).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Optional, Tuple, Set

import cv2
import numpy as np

# Importación segura de PyTorch
try:
    import torch
except ImportError:
    torch = None

# Importación segura y condicional de watchdog
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    FileSystemEventHandler = object  # Clase base simulada
    Observer = None
    WATCHDOG_AVAILABLE = False


# ============================================================
# CONFIGURACIÓN PREDETERMINADA
# ============================================================

INPUT_DIR = Path("outputs/crops_vehiculos")
PROCESSED_DIR_NAME = "procesados"
FAILED_DIR_NAME = "fallidos"

RESULTS_FILE = Path("outputs/clasificaciones_vehiculos.jsonl")
MODEL_PATH = Path("weights/vehicle_classifier")   # directorio del modelo (Transformers)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# Patrón de archivos de entrada (contrato con el productor):
#   vehiculo_<track_id>.<ext>            (formato original)
#   vehiculo_<track_id>__<video>.<ext>   (con el video: el track_id se reinicia por video y
#                                         sin él dos videos distintos colisionarían)
TRACK_PATTERN = re.compile(
    r"^vehiculo_(\d+)(?:__([A-Za-z0-9._-]+))?$",
    re.IGNORECASE
)

CONFIDENCE_THRESHOLD = 0.0

# Comprobación de estabilidad de archivos en Windows
FILE_STABLE_INTERVAL = 0.25
FILE_STABLE_CHECKS = 3
FILE_STABILITY_TIMEOUT = 10.0

MAX_RETRIES = 3
RETRY_DELAY = 1.0

QUEUE_MAX_SIZE = 1000

# Reintentos de la escritura en el JSONL una vez que el archivo ya fue movido.
APPEND_RETRIES = 3


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(threadName)s | "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("clasificador_offline")


# ============================================================
# RESULTADO DE CLASIFICACIÓN (ESQUEMA INMUTABLE)
# ============================================================

@dataclass(frozen=True)
class ClassificationResult:
    track_id: int
    imagen: str
    imagen_procesada: str
    etiqueta: str
    marca: Optional[str]
    modelo: Optional[str]
    confianza: float
    estado: str
    fingerprint_sha256: str
    timestamp_utc: str
    # Video de origen (viaja en el nombre del recorte); None con el nombre antiguo.
    video: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False
        )


# ============================================================
# CATÁLOGO DE MARCAS Y PARSER DE ETIQUETAS
# ============================================================

# Marcas compuestas ordenadas primero para evitar colisiones de prefijo
KNOWN_BRANDS = [
    # Marcas compuestas
    "mercedes_benz",
    "mercedes-benz",
    "mercedes benz",
    "alfa_romeo",
    "alfa-romeo",
    "alfa romeo",
    "aston_martin",
    "aston-martin",
    "aston martin",
    "land_rover",
    "land-rover",
    "land rover",
    "rolls_royce",
    "rolls-royce",
    "rolls royce",
    "general_motors",
    "general-motors",
    "general motors",
    # Marcas simples estándar en Argentina y mercado internacional
    "volkswagen",
    "chevrolet",
    "toyota",
    "ford",
    "fiat",
    "renault",
    "peugeot",
    "citroen",
    "honda",
    "nissan",
    "jeep",
    "hyundai",
    "kia",
    "audi",
    "bmw",
    "ram",
    "chery",
    "byd",
    "geely",
    "volvo",
    "suzuki",
    "mitsubishi",
    "subaru",
    "mazda",
    "dodge",
    "chrysler",
    "seat",
    "porsche",
    "ds",
    "iveco",
    "scania",
]

CANONICAL_BRAND_NAMES = {
    "mercedes_benz": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "alfa_romeo": "Alfa Romeo",
    "alfa-romeo": "Alfa Romeo",
    "alfa romeo": "Alfa Romeo",
    "aston_martin": "Aston Martin",
    "aston-martin": "Aston Martin",
    "aston martin": "Aston Martin",
    "land_rover": "Land Rover",
    "land-rover": "Land Rover",
    "land rover": "Land Rover",
    "rolls_royce": "Rolls-Royce",
    "rolls-royce": "Rolls-Royce",
    "rolls royce": "Rolls-Royce",
    "general_motors": "General Motors",
    "general-motors": "General Motors",
    "general motors": "General Motors",
    "volkswagen": "Volkswagen",
    "chevrolet": "Chevrolet",
    "toyota": "Toyota",
    "ford": "Ford",
    "fiat": "Fiat",
    "renault": "Renault",
    "peugeot": "Peugeot",
    "citroen": "Citroen",
    "honda": "Honda",
    "nissan": "Nissan",
    "jeep": "Jeep",
    "hyundai": "Hyundai",
    "kia": "Kia",
    "audi": "Audi",
    "bmw": "BMW",
    "ram": "RAM",
    "chery": "Chery",
    "byd": "BYD",
    "geely": "Geely",
    "volvo": "Volvo",
    "suzuki": "Suzuki",
    "mitsubishi": "Mitsubishi",
    "subaru": "Subaru",
    "mazda": "Mazda",
    "dodge": "Dodge",
    "chrysler": "Chrysler",
    "seat": "SEAT",
    "porsche": "Porsche",
    "ds": "DS",
    "iveco": "Iveco",
    "scania": "Scania",
}


def _fold(text: str) -> str:
    """Minúsculas y sin tildes/diéresis, para comparar con el catálogo ('Citroën' == 'citroen')."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _canon_key(brand: str) -> str:
    return _fold(brand).replace("-", "_").replace(" ", "_")


def parse_brand_model(label: str) -> tuple[Optional[str], Optional[str]]:
    """
    Interpreta y descompone etiquetas de clasificación en tuplas (marca, modelo).
    
    Formatos y reglas soportadas:
    1. Separadores explícitos de alta prioridad:
       - 'Toyota::Corolla' -> ('Toyota', 'Corolla')
       - 'Mercedes-Benz|C200' -> ('Mercedes-Benz', 'C200')
       - 'Ford/F_150' -> ('Ford', 'F_150')
    2. Marcas del catálogo reconocidas (incluyendo compuestas):
       - 'Ford_F_150' -> ('Ford', 'F_150')
       - 'Volkswagen_Gol_Trend' -> ('Volkswagen', 'Gol_Trend')
       - 'Mercedes_Benz_C200' -> ('Mercedes-Benz', 'C200')
       - 'Alfa_Romeo_Giulia' -> ('Alfa Romeo', 'Giulia')
    3. Marcas simples no catalogadas con exactamente un separador:
       - 'MarcaX_ModeloY' -> ('MarcaX', 'ModeloY')
       
    Casos ambiguos (marcas desconocidas con múltiples guiones) o etiquetas vacías
    devuelven (None, None) de manera controlada sin arrojar excepciones.
    """
    if not label or not isinstance(label, str):
        return None, None

    # NFC: cada letra con tilde es UN carácter, así el largo coincide con la versión sin tildes.
    raw = unicodedata.normalize("NFC", label.strip())
    if not raw:
        return None, None

    # 1. Separadores explícitos inequívocos
    for sep in ("::", "|", "/"):
        if sep in raw:
            parts = raw.split(sep, 1)
            b = parts[0].strip() or None
            m = parts[1].strip() or None
            if b:
                b_canon = CANONICAL_BRAND_NAMES.get(_canon_key(b), b)
            else:
                b_canon = None
            return b_canon, m

    raw_lower = _fold(raw)

    # 2. Búsqueda contra catálogo canónico (marcas compuestas primero)
    for brand_key in KNOWN_BRANDS:
        for delimiter in ("_", "-", " "):
            prefix = brand_key + delimiter
            if raw_lower.startswith(prefix):
                brand_canonical = CANONICAL_BRAND_NAMES.get(brand_key, brand_key.title())
                model_part = raw[len(prefix):].strip()
                return brand_canonical, (model_part or None)

    # 2b. La etiqueta ES una marca del catálogo (clasificador por marca): sin modelo.
    #     Debe ir antes del fallback, que partiría 'Mercedes-Benz' en ('Mercedes', 'Benz').
    bare = CANONICAL_BRAND_NAMES.get(_canon_key(raw))
    if bare is not None:
        return bare, None

    # 3. Fallback: exactamente un separador ('_' o '-')
    for delimiter in ("_", "-"):
        if raw.count(delimiter) == 1:
            b, m = raw.split(delimiter, 1)
            b = b.strip() or None
            m = m.strip() or None
            if b:
                b_canon = CANONICAL_BRAND_NAMES.get(_canon_key(b), b)
            else:
                b_canon = None
            return b_canon, m

    # 4. Caso no separable o ambiguo
    return None, None


# ============================================================
# UTILIDADES DE SISTEMA DE ARCHIVOS Y HASHING
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_track_id(path: Path) -> int:
    """Extrae el track_id entero desde un nombre como vehiculo_17.jpg."""
    match = TRACK_PATTERN.match(path.stem)
    if not match:
        raise ValueError(
            f"Nombre de archivo inválido: {path.name}. "
            "Se esperaba vehiculo_<track_id>.<ext>"
        )
    return int(match.group(1))


def extract_video(path: Path):
    """Nombre del video si el archivo es vehiculo_<track_id>__<video>.<ext>; None si es el formato antiguo."""
    match = TRACK_PATTERN.match(path.stem)
    return match.group(2) if match else None


def is_candidate(path: Path, input_dir: Path) -> bool:
    """
    Verifica si el archivo es un candidato legítimo para procesamiento:
    1. Debe ser un archivo existente.
    2. Debe residir directamente en input_dir (no en procesados/ o fallidos/).
    3. Debe tener una extensión de imagen soportada.
    4. NO debe ser un archivo temporal ni de sistema (.tmp, .part, oculto).
    5. Debe cumplir con el patrón vehiculo_<track_id>.<ext>.
    """
    if not path.is_file():
        return False

    name = path.name
    # Ignorar archivos temporales de escritura
    if (
        name.startswith(".")
        or name.startswith("~")
        or name.endswith(".tmp")
        or name.endswith(".part")
    ):
        return False

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    try:
        if path.resolve().parent != input_dir.resolve():
            return False
    except OSError:
        return False

    return bool(TRACK_PATTERN.match(path.stem))


def file_sha256(path: Path) -> str:
    """Calcula el hash SHA-256 del contenido binario de un archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def wait_until_file_is_stable(
    path: Path,
    stop_event: threading.Event,
    timeout: float = FILE_STABILITY_TIMEOUT,
    interval: float = FILE_STABLE_INTERVAL,
    required_stable_checks: int = FILE_STABLE_CHECKS,
) -> bool:
    """
    Verifica que el archivo esté completamente escrito y legible en Windows:
    1. Espera a que el tamaño permanezca constante durante checks consecutivos.
    2. Atrapa PermissionError transitorios causados por bloqueos de escritura.
    3. Verifica que OpenCV pueda decodificar la imagen exitosamente.
    """
    deadline = time.monotonic() + timeout
    previous_size = None
    stable_checks = 0

    while time.monotonic() < deadline:
        if stop_event.is_set():
            return False

        if not path.exists():
            stable_checks = 0
            previous_size = None
            stop_event.wait(interval)
            continue

        try:
            current_size = path.stat().st_size
        except (PermissionError, FileNotFoundError, OSError):
            # En Windows, un archivo en proceso de escritura arroja WinError 32
            stable_checks = 0
            previous_size = None
            stop_event.wait(interval)
            continue

        if current_size > 0 and current_size == previous_size:
            stable_checks += 1
        else:
            stable_checks = 0

        previous_size = current_size

        if stable_checks >= required_stable_checks:
            try:
                # np.fromfile + imdecode en vez de cv2.imread: imread falla en Windows con rutas
                # que tienen tildes o ñ (p. ej. carpetas como 'Exportación').
                image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if (
                    image is not None
                    and image.size > 0
                    and image.shape[0] > 0
                    and image.shape[1] > 0
                ):
                    return True
            except Exception as exc:
                logger.debug("cv2.imread falló transitoriamente en %s: %s", path.name, exc)
            stable_checks = 0

        stop_event.wait(interval)

    return False


# ============================================================
# ESCRITOR JSONL CON CONTROL DE CONCURRENCIA
# ============================================================

class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_trailing_newline(self) -> None:
        """Si una escritura anterior quedó cortada (sin '\\n' final), el próximo registro se
        pegaría a esa línea y se perderían los dos. Se cierra la línea parcial antes de anexar."""
        try:
            if not self.path.exists() or self.path.stat().st_size == 0:
                return
            with self.path.open("rb") as file:
                file.seek(-1, os.SEEK_END)
                last = file.read(1)
            if last != b"\n":
                with self.path.open("ab") as file:
                    file.write(b"\n")
                    file.flush()
                    os.fsync(file.fileno())
        except OSError as exc:
            logger.error("No se pudo verificar el final de %s: %s", self.path, exc)

    def append(self, result: ClassificationResult) -> None:
        line = result.to_json() + "\n"
        with self.lock:
            self._ensure_trailing_newline()
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line)
                file.flush()
                os.fsync(file.fileno())

    def load_processed_hashes(self, include_mock: bool = True) -> set[str]:
        """Huellas ya procesadas. `include_mock=False` ignora las lecturas simuladas, para que un
        recorte probado con --mock no cuente como 'ya clasificado' ante el modelo real."""
        accepted = {"ok", "below_threshold"} | ({"mock"} if include_mock else set())
        processed = set()
        if not self.path.exists():
            return processed

        try:
            with self.path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    fingerprint = row.get("fingerprint_sha256")
                    status = row.get("estado")

                    if fingerprint and status in accepted:
                        processed.add(fingerprint)
        except OSError as exc:
            logger.error("No se pudo leer %s: %s", self.path, exc)

        return processed


# ============================================================
# CLASIFICADOR (TRANSFORMERS, REAL O SIMULADO)
# ============================================================

class VehicleClassifier:
    """Clasifica marca y modelo con un modelo de imágenes de Hugging Face Transformers (Apache-2.0).

    Reemplaza a Ultralytics YOLO-cls: Ultralytics se distribuye bajo AGPL-3.0 (copyleft), que obliga a
    publicar el código derivado o a comprar una licencia comercial. Ver docs/LICENCIAS.md.

    El modelo es un DIRECTORIO (igual que el RT-DETR de patentes): `config.json` con `id2label`,
    los pesos y `preprocessor_config.json`. Los nombres de clase deben tener la forma `Marca_Modelo`.
    """

    def __init__(
        self,
        model_path: Path,
        device: str,
        mock: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.mock = mock
        self.model = None
        self.processor = None
        self._torch = None

        if self.mock:
            logger.warning(
                "MODO MOCK ACTIVADO. No se cargará ni utilizará un modelo de red neuronal."
            )
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo de clasificación: {self.model_path}\n"
                "Es un directorio con config.json, los pesos y preprocessor_config.json.\n"
                "Para operar sin modelo entrenado, utilizá el argumento --mock."
            )

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError(
                "Faltan torch o transformers en este entorno.\n"
                "Instalá con: pip install torch transformers"
            ) from exc

        logger.info("Cargando modelo de clasificación desde: %s", self.model_path)
        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        self.model = (
            AutoModelForImageClassification.from_pretrained(str(self.model_path))
            .to(self.device)
            .eval()
        )
        logger.info("Modelo cargado exitosamente en dispositivo: %s", self.device)

    def classify(
        self,
        image_path: Path,
    ) -> tuple[str, Optional[str], Optional[str], float]:
        if self.mock:
            # En modo mock no pretendemos que exista un modelo ni inventamos predicciones reales
            return ("MOCK_VEHICULO", None, None, 0.0)

        if self.model is None or self.processor is None:
            raise RuntimeError("No hay un modelo de clasificación cargado.")

        # np.fromfile + imdecode: funciona con rutas con tildes en Windows (cv2.imread no).
        bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"No se pudo decodificar la imagen: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probs = self._torch.softmax(logits.float(), dim=-1)
            top_prob, top_index = probs.max(dim=-1)
            class_id = int(top_index)
            confidence = float(top_prob)

        label = str(self.model.config.id2label[class_id])
        brand, model = parse_brand_model(label)

        return (label, brand, model, confidence)


# ============================================================
# OBSERVADORES DE SISTEMA DE ARCHIVOS (WATCHDOG Y POLLING)
# ============================================================

class VehicleFileHandler(FileSystemEventHandler):
    """Manejador de eventos nativos del sistema de archivos mediante watchdog."""
    def __init__(self, processor: FileProcessor, input_dir: Path):
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir

    def on_created(self, event):
        if not event.is_directory:
            self.processor.enqueue(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self.processor.enqueue(Path(event.dest_path))


class PollingWatcher:
    """Observador periódico de respaldo cuando watchdog no está instalado."""
    def __init__(self, processor: FileProcessor, input_dir: Path, poll_interval: float = 0.5):
        self.processor = processor
        self.input_dir = input_dir
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._poll_loop,
            name="polling-watcher",
            daemon=True,
        )

    def _poll_loop(self):
        while not self.stop_event.is_set():
            try:
                self.processor.scan_pending()
            except Exception as exc:
                logger.error("Error en sondeo periódico de directorio: %s", exc)
            self.stop_event.wait(self.poll_interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def join(self, timeout=None):
        self.thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self.thread.is_alive()


# ============================================================
# PROCESADOR IDEMPOTENTE Y SEGURO PARA WINDOWS
# ============================================================

class FileProcessor:
    def __init__(
        self,
        input_dir: Path,
        classifier: VehicleClassifier,
        writer: JsonlWriter,
        stop_event: threading.Event,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY,
    ):
        self.input_dir = input_dir.resolve()
        self.processed_dir = self.input_dir / PROCESSED_DIR_NAME
        self.failed_dir = self.input_dir / FAILED_DIR_NAME

        self.classifier = classifier
        self.writer = writer
        self.stop_event = stop_event
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.pending: Set[Path] = set()
        self.pending_lock = threading.Lock()

        # Un clasificador real no debe dar por procesado lo que solo pasó por --mock.
        self.processed_hashes = writer.load_processed_hashes(include_mock=bool(classifier.mock))
        self.processed_lock = threading.Lock()

        self.worker = threading.Thread(
            target=self.worker_loop,
            name="classifier-worker",
            daemon=True,
        )

    def start(self):
        self.worker.start()

    def enqueue(self, path: Path):
        if not is_candidate(path, self.input_dir):
            return

        resolved = path.resolve()
        with self.pending_lock:
            if resolved in self.pending:
                return
            self.pending.add(resolved)

        try:
            self.queue.put_nowait(resolved)
            logger.info("Recorte encolado: %s", resolved.name)
        except Exception:
            with self.pending_lock:
                self.pending.discard(resolved)
            logger.exception("Error al encolar archivo: %s", resolved)

    def scan_pending(self):
        if not self.input_dir.exists():
            return

        try:
            for path in self.input_dir.iterdir():
                self.enqueue(path)
        except OSError as exc:
            logger.error("Error al escanear directorio %s: %s", self.input_dir, exc)

    def worker_loop(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                path = self.queue.get(timeout=0.25)
            except Empty:
                continue

            try:
                self.process_file(path)
            finally:
                with self.pending_lock:
                    self.pending.discard(path.resolve())
                self.queue.task_done()

    def process_file(self, path: Path):
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if not path.exists():
                    logger.warning("El archivo no existe o fue movido: %s", path)
                    return

                stable = wait_until_file_is_stable(path, self.stop_event)
                if not stable and self.stop_event.is_set():
                    # Cierre solicitado: no es un fallo del archivo. Se deja en la carpeta de
                    # entrada para que lo procese el próximo arranque (antes iba a 'fallidos').
                    logger.info("Cierre solicitado: %s queda pendiente, sin procesar.", path.name)
                    return
                if not stable:
                    raise TimeoutError(
                        f"El archivo {path.name} no alcanzó un estado legible y estable tras {FILE_STABILITY_TIMEOUT}s."
                    )

                fingerprint = file_sha256(path)

                # Detección temprana de duplicados por hash
                with self.processed_lock:
                    already_processed = fingerprint in self.processed_hashes

                if already_processed:
                    logger.info("Crop ya procesado previamente (hash conocido): %s", path.name)
                    self.move_file(path, self.processed_dir)
                    return

                track_id = extract_track_id(path)
                video = extract_video(path)

                logger.info(
                    "Clasificando track_id=%d (intento %d/%d)",
                    track_id, attempt, self.max_retries
                )

                label, brand, model, confidence = self.classifier.classify(path)

                if self.classifier.mock:
                    status = "mock"
                elif confidence >= CONFIDENCE_THRESHOLD:
                    status = "ok"
                else:
                    status = "below_threshold"

                # 1. Movimiento físico PRIMERO (garantiza ubicación segura)
                actual_destination = self.move_file(path, self.processed_dir)

                # 2. Persistencia en JSONL con la ruta física exacta
                result = ClassificationResult(
                    track_id=track_id,
                    imagen=str(path),
                    imagen_procesada=str(actual_destination),
                    etiqueta=label,
                    marca=brand,
                    modelo=model,
                    confianza=confidence,
                    estado=status,
                    fingerprint_sha256=fingerprint,
                    timestamp_utc=utc_now(),
                    video=video,
                )

                if self._persist(result):
                    with self.processed_lock:
                        self.processed_hashes.add(fingerprint)

                logger.info(
                    "Clasificación terminada: track=%d | %s | %.3f -> %s",
                    track_id, label, confidence, actual_destination.name
                )
                return

            except Exception as exc:
                last_exc = exc
                logger.exception(
                    "Error procesando %s (intento %d/%d): %s",
                    path.name, attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    self.stop_event.wait(self.retry_delay)

        # Si se agotaron todos los reintentos: registrar fallo transaccional
        self.write_error(path, last_exc)

    def _persist(self, result: ClassificationResult) -> bool:
        """Anexa el registro al JSONL con reintentos. El archivo ya fue movido, así que un fallo
        aquí no puede reintentar la clasificación (el recorte ya no está): si no se logra
        persistir, el registro completo queda en el log para recuperarlo a mano."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, APPEND_RETRIES + 1):
            try:
                self.writer.append(result)
                return True
            except Exception as exc:
                last_exc = exc
                logger.error("No se pudo escribir el JSONL (intento %d/%d): %s",
                             attempt, APPEND_RETRIES, exc)
                self.stop_event.wait(self.retry_delay)
        logger.error("REGISTRO NO PERSISTIDO (%s). Recuperar manualmente: %s", last_exc, result.to_json())
        return False

    def write_error(self, path: Path, error: Optional[Exception]):
        try:
            track_id = extract_track_id(path)
            video = extract_video(path)
        except Exception:
            track_id = -1
            video = None

        fingerprint = ""
        if path.exists():
            try:
                fingerprint = file_sha256(path)
            except OSError:
                pass

        actual_dest = self.failed_dir / path.name
        if path.exists():
            try:
                actual_dest = self.move_file(path, self.failed_dir)
            except Exception as exc:
                logger.error(
                    "No se pudo mover el archivo fallido %s a %s: %s",
                    path, self.failed_dir, exc
                )

        result = ClassificationResult(
            track_id=track_id,
            imagen=str(path),
            imagen_procesada=str(actual_dest),
            etiqueta="",
            marca=None,
            modelo=None,
            confianza=0.0,
            estado=f"error: {error}",
            fingerprint_sha256=fingerprint,
            timestamp_utc=utc_now(),
            video=video,
        )

        self._persist(result)

    @staticmethod
    def move_file(
        source: Path,
        destination_dir: Path,
        max_retries: int = 5,
        initial_delay: float = 0.1,
    ) -> Path:
        """
        Mueve un archivo evitando sobrescrituras accidentales y con reintentos
        ante bloqueos transitorios típicos de Windows (WinError 32).
        Devuelve el Path definitivo donde se alojó el archivo.
        """
        if not source.exists():
            raise FileNotFoundError(f"Archivo origen no encontrado: {source}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name

        # Resolución de colisiones sin sobrescritura
        if destination.exists():
            index = 1
            while True:
                alternative = destination_dir / f"{source.stem}_{index}{source.suffix}"
                if not alternative.exists():
                    destination = alternative
                    break
                index += 1

        delay = initial_delay
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                shutil.move(str(source), str(destination))
                return destination
            except (PermissionError, OSError) as exc:
                last_exc = exc
                time.sleep(delay)
                delay *= 1.5

        raise OSError(
            f"Fallo definitivo al mover {source.name} a {destination} tras {max_retries} intentos: {last_exc}"
        )


# ============================================================
# RESOLUCIÓN DE DISPOSITIVO E INFERENCIA
# ============================================================

def resolve_device(requested: str, mock: bool = False) -> str:
    if mock:
        return "cpu"

    requested = requested.lower()
    if requested == "cpu":
        return "cpu"

    if requested in {"cuda", "cuda:0"}:
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    return "cuda:0"
            except Exception:
                pass
        logger.warning("CUDA no disponible en PyTorch. Se utilizará CPU.")
        return "cpu"

    return requested


# ============================================================
# PARSER DE LÍNEA DE COMANDOS
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Clasificador offline de vehículos (Consumidor Desacoplado)."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Directorio de entrada de recortes de vehículos. Default: {INPUT_DIR}",
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_FILE,
        help=f"Archivo JSONL de salida para clasificaciones. Default: {RESULTS_FILE}",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Directorio del modelo de clasificación (Transformers: config.json + pesos). Default: {MODEL_PATH}",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cuda:0", "cpu"],
        help="Dispositivo de inferencia para el modelo.",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Modo simulado: procesa y valida flujo sin cargar pesos de red neuronal.",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Intervalo de sondeo (segundos) cuando se usa PollingWatcher. Default: 0.5",
    )

    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Procesa los archivos pendientes y finaliza sin esperar nuevos eventos.",
    )

    return parser.parse_args(argv)


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================

def main(argv=None):
    args = parse_args(argv)

    input_dir = args.input_dir.resolve()
    processed_dir = input_dir / PROCESSED_DIR_NAME
    failed_dir = input_dir / FAILED_DIR_NAME
    results_file = args.results_file.resolve()

    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, mock=args.mock)

    logger.info("========================================")
    logger.info("CLASIFICADOR OFFLINE DE VEHÍCULOS")
    logger.info("========================================")
    logger.info("Entrada:   %s", input_dir)
    logger.info("Salida:    %s", results_file)
    logger.info("Modelo:    %s", args.model.resolve())
    logger.info("Device:    %s", device)
    logger.info("Modo Mock: %s", args.mock)
    logger.info("Motor FS:  %s", "watchdog" if WATCHDOG_AVAILABLE else "PollingWatcher (fallback)")

    classifier = VehicleClassifier(
        model_path=args.model,
        device=device,
        mock=args.mock,
    )

    writer = JsonlWriter(results_file)
    stop_event = threading.Event()

    processor = FileProcessor(
        input_dir=input_dir,
        classifier=classifier,
        writer=writer,
        stop_event=stop_event,
    )

    # Selección desacoplada del motor de observación de archivos
    if WATCHDOG_AVAILABLE:
        handler = VehicleFileHandler(processor, input_dir)
        watcher = Observer()
        watcher.schedule(handler, str(input_dir), recursive=False)
    else:
        watcher = PollingWatcher(
            processor,
            input_dir,
            poll_interval=args.poll_interval,
        )

    # Iniciar motor y worker
    watcher.start()
    processor.start()

    # Escaneo inicial de archivos pendientes
    processor.scan_pending()

    if args.single_run:
        logger.info("Modo --single-run activo. Esperando a procesar la cola...")
        processor.queue.join()
        logger.info("Cola vacía. Finalizando consumidor en modo batch.")
        stop_event.set()
        watcher.stop()
        watcher.join(timeout=3.0)
        return

    logger.info("Consumidor iniciado. Esperando nuevos recortes...")

    try:
        while watcher.is_alive():
            watcher.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Cierre solicitado por el usuario.")
    finally:
        watcher.stop()
        stop_event.set()
        watcher.join(timeout=5.0)
        processor.queue.join()
        logger.info("Clasificador offline finalizado limpiamente.")


if __name__ == "__main__":
    main()