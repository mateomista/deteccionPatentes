"""Observadores del directorio de entrada: watchdog (si está instalado) o sondeo periódico."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

# Importación segura y condicional de watchdog
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    FileSystemEventHandler = object  # Clase base simulada
    Observer = None
    WATCHDOG_AVAILABLE = False

if TYPE_CHECKING:
    from .processor import FileProcessor

logger = logging.getLogger("clasificador_offline")


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
