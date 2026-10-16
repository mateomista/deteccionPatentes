"""Configuración predeterminada del clasificador offline de vehículos."""
import re
from pathlib import Path

INPUT_DIR = Path("outputs/crops_vehiculos")
PROCESSED_DIR_NAME = "procesados"
FAILED_DIR_NAME = "fallidos"

RESULTS_FILE = Path("outputs/clasificaciones_vehiculos.jsonl")
MODEL_PATH = Path("weights/vehicle_classifier.pt")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# Patrón canónico de archivos de entrada: vehiculo_<track_id>.<ext>
TRACK_PATTERN = re.compile(
    r"^vehiculo_(\d+)$",
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
