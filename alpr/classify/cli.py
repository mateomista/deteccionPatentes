"""Línea de comandos del clasificador offline: python -m alpr.classify"""
import argparse
import logging
import threading
from pathlib import Path

# Importación segura de PyTorch
try:
    import torch
except ImportError:
    torch = None

from .classifier import VehicleClassifier
from .config import (
    FAILED_DIR_NAME, INPUT_DIR, MODEL_PATH, PROCESSED_DIR_NAME, RESULTS_FILE,
)
from .processor import FileProcessor
from .results import JsonlWriter
from .watchers import Observer, PollingWatcher, VehicleFileHandler, WATCHDOG_AVAILABLE

logger = logging.getLogger("clasificador_offline")


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
        help=f"Ruta al peso YOLO-cls fine-tuneado. Default: {MODEL_PATH}",
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


def setup_logging():
    """Misma configuración que el script original, ahora al arrancar y no al importar."""
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


def main(argv=None):
    args = parse_args(argv)
    setup_logging()

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
