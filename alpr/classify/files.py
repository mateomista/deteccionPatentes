"""Utilidades de sistema de archivos: candidatos, hash y estabilidad de archivos (Windows)."""
import hashlib
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import (
    FILE_STABILITY_TIMEOUT, FILE_STABLE_CHECKS, FILE_STABLE_INTERVAL,
    SUPPORTED_EXTENSIONS, TRACK_PATTERN,
)

logger = logging.getLogger("clasificador_offline")


def extract_track_id(path: Path) -> int:
    """Extrae el track_id entero desde un nombre como vehiculo_17.jpg."""
    match = TRACK_PATTERN.match(path.stem)
    if not match:
        raise ValueError(
            f"Nombre de archivo inválido: {path.name}. "
            "Se esperaba vehiculo_<track_id>.<ext>"
        )
    return int(match.group(1))


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
