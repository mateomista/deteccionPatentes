"""CLI de PRODUCCIÓN / inferencia: `python -m alpr.inference --source video.mp4`.

Los umbrales salen de (en orden de prioridad): flags --confidence-* / --min-* ...,
luego `--thresholds archivo.json` (p. ej. el best_thresholds.json de la
calibración), luego los valores por defecto de `Thresholds`.
"""
import argparse
import logging
import os

from .config import DEFAULT_RTDETR_BATCH, OCR_MAX_ATTEMPTS, FLUSH_UNFINALIZED_DEFAULT, \
    STRICT_PLATE_VALIDATION, PipelineOptions, Thresholds

THRESHOLD_FLAGS = {
    "confidence_vehicle": float, "confidence_plate": float, "min_votes": int,
    "min_avg_ocr_conf": float, "early_finalize_votes": int, "early_finalize_conf": float,
    "min_consensus_margin": float, "flush_single_vote_conf": float,
}


def add_pipeline_args(p):
    """Flags compartidos por inferencia, calibración y evaluación."""
    p.add_argument("--plate-model", default=os.path.join("outputs", "rtdetr-ccpd", "final_model"))
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dual-ocr", action=argparse.BooleanOptionalAction, default=True,
                   help="Habilita las variantes de rescate del OCR; --no-dual-ocr deja solo el intento base")
    p.add_argument("--ocr-max-attempts", type=int, default=OCR_MAX_ATTEMPTS)
    p.add_argument("--disable-focus-mask", action="store_true")
    p.add_argument("--rtdetr-batch", type=int, default=DEFAULT_RTDETR_BATCH)
    p.add_argument("--sequential-rtdetr", action="store_true")
    p.add_argument("--strict-plate", action=argparse.BooleanOptionalAction,
                   default=STRICT_PLATE_VALIDATION,
                   help="Validación ESTRICTA (LLLNNN/LLNNNLL/LNNNLLL). Usar en producción y en "
                        "toda calibración/evaluación que se vaya a citar")
    p.add_argument("--flush-unfinalized", action=argparse.BooleanOptionalAction,
                   default=FLUSH_UNFINALIZED_DEFAULT)
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--dedup-seconds", type=float, default=30.0)
    p.add_argument("--dedup-max-distance", type=int, choices=(0, 1, 2), default=0,
                   help="Deduplicación aproximada: patentes que difieren en a lo sumo N caracteres "
                        "dentro de --dedup-seconds cuentan como la misma. 0 = solo exacta (por defecto). "
                        "Riesgo: puede fusionar dos vehículos con patentes casi iguales")
    p.add_argument("--char-voting", action="store_true",
                   help="Consenso carácter por carácter entre las lecturas de cada vehículo, en vez "
                        "de exigir lecturas idénticas. Por defecto apagado: evaluar con ground truth")
    p.add_argument("--max-frames", type=int, default=0, help="0 = video completo")


def add_threshold_args(p):
    p.add_argument("--thresholds", default=None,
                   help="JSON de umbrales (p. ej. outputs/calibration/best_thresholds.json)")
    for name, typ in THRESHOLD_FLAGS.items():
        p.add_argument("--" + name.replace("_", "-"), type=typ, default=None,
                       help=f"Override de Thresholds.{name}")


def thresholds_from_args(args):
    th = Thresholds.from_json(args.thresholds) if getattr(args, "thresholds", None) else Thresholds()
    overrides = {n: getattr(args, n) for n in THRESHOLD_FLAGS if getattr(args, n, None) is not None}
    return th.replace(**overrides) if overrides else th


def options_from_args(args, snapshots_dir="snapshots", debug_crops=False, export_vehicle_dir=None):
    return PipelineOptions(
        plate_model=args.plate_model, snapshots_dir=snapshots_dir, max_frames=args.max_frames,
        dedup_seconds=args.dedup_seconds, dedup_max_distance=getattr(args, "dedup_max_distance", 0),
        char_voting=getattr(args, "char_voting", False), fp16=args.fp16, dual_ocr=args.dual_ocr,
        ocr_max_attempts=args.ocr_max_attempts, disable_focus_mask=args.disable_focus_mask,
        debug_crops=debug_crops, rtdetr_batch=args.rtdetr_batch,
        sequential_rtdetr=args.sequential_rtdetr, strict_plate=args.strict_plate,
        flush_unfinalized=args.flush_unfinalized, cudnn_benchmark=args.cudnn_benchmark,
        export_vehicle_dir=export_vehicle_dir)


def main(argv=None):
    p = argparse.ArgumentParser(description="ALPR — inferencia sobre un video")
    p.add_argument("--source", default=os.path.join("inputs", "trafico.mp4"))
    p.add_argument("--snapshots-dir", default="snapshots")
    p.add_argument("--output-json", default="detecciones_alpr.json")
    p.add_argument("--output-video", default=os.path.join("outputs", "alpr.mp4"))
    p.add_argument("--debug-crops", action="store_true")
    p.add_argument("--export-vehicle-crops", default=None, metavar="DIR",
                   help="Exporta el recorte del vehículo de cada patente registrada (contrato con "
                        "alpr.classify: vehiculo_<track_id>__<video>.jpg). Por defecto no exporta")
    add_pipeline_args(p)
    add_threshold_args(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
    th = thresholds_from_args(args)
    if not os.path.exists(args.plate_model):
        raise FileNotFoundError(f"El modelo RT-DETR no existe: {args.plate_model}")
    if not args.strict_plate:
        logging.warning("[POLÍTICA] Validación PERMISIVA/DEBUG: NO equivale a formato argentino. "
                        "Para producción usar --strict-plate.")
    logging.info("[POLÍTICA] Umbrales: %s", th.to_dict())

    from .models import ModelBundle
    from .pipeline import ALPRPipeline
    models = ModelBundle(args.plate_model, cudnn_benchmark=args.cudnn_benchmark)
    pipeline = ALPRPipeline(models, options_from_args(args, args.snapshots_dir, args.debug_crops,
                                                        args.export_vehicle_crops), th)
    pipeline.run_video(args.source, output_json=args.output_json, output_video=args.output_video)


if __name__ == "__main__":
    main()
