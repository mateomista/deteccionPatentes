"""Configuración predeterminada del clasificador offline de vehículos."""
from pathlib import Path

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

# Patrón de archivos de entrada, definido por el contrato con el productor (alpr/contracts.py):
#   vehiculo_<track_id>.<ext>   (formato original)
#   vehiculo_<track_id>__<video>.<ext>   (con el video, para no colisionar entre videos)
from alpr.contracts import VEHICLE_CROP_PATTERN

TRACK_PATTERN = VEHICLE_CROP_PATTERN

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
