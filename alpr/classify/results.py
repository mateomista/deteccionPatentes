"""Esquema de resultado y escritor JSONL con control de concurrencia."""
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("clasificador_offline")


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
    # La clave de correlación con las detecciones del pipeline es (video, track_id).
    video: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



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
