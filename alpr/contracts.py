"""Contrato de intercambio entre el productor (pipeline) y el consumidor (alpr.classify).

El pipeline escribe, por cada vehículo cuya patente quedó registrada, UN recorte del
vehículo completo en una carpeta; el clasificador la observa y lo procesa.

    nombre:   vehiculo_<track_id>__<video>.<ext>      p. ej. vehiculo_17__trafico01.jpg
    escritura: atómica (<nombre>.tmp -> renombrar), para que el consumidor nunca vea un
               archivo a medio escribir (ignora los terminados en .tmp).

POR QUÉ EL VIDEO VA EN EL NOMBRE: el `track_id` lo asigna ByteTrack y se reinicia en cada
video. Sin el video, `vehiculo_1.jpg` de dos videos distintos colisionaría y la clasificación
no se podría asociar a la patente correcta. La clave de correlación es (video, track_id),
que coincide con (nombre del video, `id_vehiculo`) de las detecciones del pipeline.

Compatibilidad: el formato original `vehiculo_<track_id>.<ext>` (sin video) sigue siendo válido.

Módulo sin dependencias pesadas (solo OpenCV para codificar la imagen).
"""
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

PREFIX = "vehiculo"
SEPARATOR = "__"
# vehiculo_<track_id>  o  vehiculo_<track_id>__<video>
VEHICLE_CROP_PATTERN = re.compile(r"^vehiculo_(\d+)(?:__([A-Za-z0-9._-]+))?$", re.IGNORECASE)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_video_stem(stem: str) -> str:
    """Deja solo caracteres seguros para un nombre de archivo (también en Windows).
    Dos videos con nombres muy parecidos pueden colapsar al mismo valor: usá nombres distintos."""
    clean = _UNSAFE.sub("-", str(stem)).strip("-._")
    return clean or "video"


def vehicle_crop_filename(track_id: int, video_stem: Optional[str] = None, ext: str = ".jpg") -> str:
    if int(track_id) < 0:
        raise ValueError(f"track_id inválido: {track_id}")
    name = f"{PREFIX}_{int(track_id)}"
    if video_stem:
        name += f"{SEPARATOR}{sanitize_video_stem(video_stem)}"
    return name + ext


def parse_vehicle_crop_stem(stem: str) -> Optional[Tuple[int, Optional[str]]]:
    """(track_id, video | None) o None si el nombre no cumple el contrato."""
    m = VEHICLE_CROP_PATTERN.match(stem)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def write_image_atomic(path, image, jpeg_quality: int = 95, retries: int = 3, delay: float = 0.1) -> Path:
    """Escribe `image` en `path` de forma atómica: primero a `<path>.tmp` y después se renombra.

    Codifica con cv2.imencode y escribe los bytes con Python: así funciona con rutas con
    tildes en Windows, donde cv2.imwrite falla. Reintenta ante bloqueos transitorios."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como {ext}")

    tmp = path.with_name(path.name + ".tmp")
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return path
        except OSError as exc:        # incluye PermissionError (WinError 32)
            last_exc = exc
            time.sleep(delay * attempt)
    try:
        tmp.unlink()
    except OSError:
        pass
    raise OSError(f"No se pudo escribir {path} tras {retries} intentos: {last_exc}")
