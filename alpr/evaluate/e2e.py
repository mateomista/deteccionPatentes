"""EVALUACIÓN end-to-end sobre la partición `test` (una sola vez, umbrales congelados).

    python -m alpr.evaluate.e2e --splits splits/videos.json --split test \
        --videos-dir data/videos --ground-truth data/ground_truth.csv \
        --thresholds outputs/calibration/best_thresholds.json --strict-plate \
        --out-dir outputs/eval_test

Corre el pipeline REAL (no el replay) sobre cada video y compara contra el GT.
Reporta aparte "todas" y "solo consenso": las recuperadas (track_lost /
end_of_video) tienen evidencia más débil y no deben mezclarse sin avisar.
"""
import argparse
import json
import logging
import os

from ..data.splits import load_split
from ..inference import add_pipeline_args, add_threshold_args, options_from_args, thresholds_from_args
from ..config import CONSENSUS_REASON, Thresholds
from .metrics import evaluate_predictions, load_ground_truth

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def find_video(videos_dir, stem):
    for ext in VIDEO_EXTS:
        path = os.path.join(videos_dir, stem + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No encontré el video {stem!r} en {videos_dir}")


def format_report(title, m):
    lines = [f"== {title} ==",
             f"  vehículos GT: {m['n_gt']} | detecciones: {m['n_pred']}",
             f"  TP={m['tp']}  patente errónea={m['wrong']}  FP puro={m['spurious']}  "
             f"duplicadas={m['duplicate']}  FN={m['fn']}",
             f"  precisión={m['precision']:.3f}  recall={m['recall']:.3f}  F1={m['f1']:.3f}"]
    if m["cer_on_read"] is not None:
        lines.append(f"  CER sobre vehículos leídos: {m['cer_on_read']:.3f}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--splits", default=os.path.join("splits", "videos.json"))
    p.add_argument("--split", default="test")
    p.add_argument("--videos-dir", default=os.path.join("data", "videos"))
    p.add_argument("--ground-truth", default=os.path.join("data", "ground_truth.csv"))
    p.add_argument("--out-dir", default=os.path.join("outputs", "eval_test"))
    p.add_argument("--tolerance-frames", type=int, default=0)
    p.add_argument("--default-thresholds", action="store_true",
                   help="Evalúa con los umbrales por defecto (línea base sin calibrar)")
    p.add_argument("--save-snapshots", action="store_true")
    add_pipeline_args(p)
    add_threshold_args(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
    if not args.thresholds and not args.default_thresholds:
        p.error("Indicá --thresholds <json> (los calibrados) o --default-thresholds de forma explícita")
    th = thresholds_from_args(args) if not args.default_thresholds else Thresholds()

    videos = load_split(args.splits, args.split)
    gt = load_ground_truth(args.ground_truth)
    missing = [v for v in videos if v not in gt]
    if missing:
        raise SystemExit(f"Videos de la partición {args.split!r} sin ground truth: {missing}")
    gt = {v: gt[v] for v in videos}

    os.makedirs(args.out_dir, exist_ok=True)
    snaps = os.path.join(args.out_dir, "snapshots") if args.save_snapshots else None

    from ..models import ModelBundle
    from ..pipeline import ALPRPipeline
    models = ModelBundle(args.plate_model, cudnn_benchmark=args.cudnn_benchmark)
    pipeline = ALPRPipeline(models, options_from_args(args, snapshots_dir=snaps), th)

    preds, runs = {}, {}
    for stem in videos:
        res = pipeline.run_video(find_video(args.videos_dir, stem),
                                 output_json=os.path.join(args.out_dir, f"pred_{stem}.json"))
        preds[stem] = res.detections
        runs[stem] = {"frames": res.frames, "seconds": round(res.elapsed, 2),
                      "fps": round(res.frames / max(res.elapsed, 1e-6), 2),
                      "audit_violations": len(res.audit_problems)}

    result = {
        "split": args.split, "thresholds": th.to_dict(), "strict_plate": args.strict_plate,
        "tolerance_frames": args.tolerance_frames, "runs": runs,
        "all": evaluate_predictions(gt, preds, args.tolerance_frames),
        "consensus_only": evaluate_predictions(gt, preds, args.tolerance_frames, {CONSENSUS_REASON}),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(format_report("TODAS las detecciones", result["all"]))
    print(format_report("SOLO consenso", result["consensus_only"]))
    print(f"-> {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
