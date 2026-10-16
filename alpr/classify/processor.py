"""Procesador idempotente de recortes, seguro para Windows (cola + worker)."""
import logging
import shutil
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Optional, Set

from .classifier import VehicleClassifier
from .config import (
    APPEND_RETRIES, CONFIDENCE_THRESHOLD, FAILED_DIR_NAME, FILE_STABILITY_TIMEOUT, MAX_RETRIES,
    PROCESSED_DIR_NAME, QUEUE_MAX_SIZE, RETRY_DELAY,
)
from .files import extract_track_id, file_sha256, is_candidate, wait_until_file_is_stable
from .results import ClassificationResult, JsonlWriter, utc_now

logger = logging.getLogger("clasificador_offline")


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
        except Exception:
            track_id = -1

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
