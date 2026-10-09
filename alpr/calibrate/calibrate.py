"""CALIBRACIÓN de umbrales sobre la partición `calib` (nunca sobre `test`).

Dos pasos, porque los umbrales son de dos clases (ver config.DETECTION_FIELDS /
DECISION_FIELDS):

  1) record  — corre los modelos UNA vez por video (y por combinación de
     umbrales de DETECCIÓN) y graba cada lectura: ~lo caro, con GPU.
         python -m alpr.calibrate.calibrate record --splits splits/videos.json \
             --split calib --videos-dir data/videos --strict-plate \
             --obs-dir outputs/calibration/obs \
             --confidence-vehicle-grid 0.35 0.45 --confidence-plate-grid 0.15 0.25 0.4

  2) sweep   — re-decide el consenso OFFLINE con una grilla de umbrales de
     DECISIÓN contra el GT, en segundos y sin GPU. Elige la mejor combinación.
         python -m alpr.calibrate.calibrate sweep --obs-dir outputs/calibration/obs \
             --ground-truth data/ground_truth.csv --out-dir outputs/calibration \
             --objective recall_at_precision --min-precision 0.95

Salida: sweep_results.csv (todas las combinaciones) y best_thresholds.json.
Después, UNA sola evaluación en test con esos umbrales congelados:
    python -m alpr.evaluate.e2e --thresholds outputs/calibration/best_thresholds.json ...
"""
import argparse
import csv
import glob
import itertools
import json
import logging
import os

from ..config import DECISION_FIELDS, Thresholds
from ..data.splits import load_manifest
from ..evaluate.metrics import evaluate_predictions, load_ground_truth
from ..replay import replay_video

DEFAULT_GRID = {
    "min_votes": [1, 2, 3],
    "min_avg_ocr_conf": [0.10, 0.20, 0.30, 0.40, 0.50],
    "min_consensus_margin": [0.0, 0.15, 0.30],
    "early_finalize_votes": [2, 3, 4],
    "early_finalize_conf": [0.80, 0.90, 0.95],
    "flush_single_vote_conf": [0.40, 0.50, 0.60, 0.70],
}


# ------------------------------------------------------------------------------
# Paso 1: grabar observaciones
# ------------------------------------------------------------------------------
def _guard_not_test(manifest, videos, allowed_forbidden=("test",)):
    """Impide calibrar con videos de la partición de test."""
    for forbidden in allowed_forbidden:
        bad = set(videos) & set(manifest["splits"].get(forbidden, []))
        if bad:
            raise SystemExit(f"CALIBRAR CON TEST INVALIDA LA EVALUACIÓN: {sorted(bad)} están en "
                             f"la partición {forbidden!r}. Usá --split calib.")


def cmd_record(args):
    from ..evaluate.e2e import find_video
    from ..inference import options_from_args, thresholds_from_args
    from ..models import ModelBundle
    from ..pipeline import ALPRPipeline

    manifest = load_manifest(args.splits)
    if args.split not in manifest["splits"]:
        raise SystemExit(f"Partición {args.split!r} inexistente en {args.splits}")
    videos = manifest["splits"][args.split]
    _guard_not_test(manifest, videos)

    base = thresholds_from_args(args)
    cv_grid = args.confidence_vehicle_grid or [base.confidence_vehicle]
    cp_grid = args.confidence_plate_grid or [base.confidence_plate]
    os.makedirs(args.obs_dir, exist_ok=True)

    models = ModelBundle(args.plate_model, cudnn_benchmark=args.cudnn_benchmark)
    for cv, cp in itertools.product(cv_grid, cp_grid):
        th = base.replace(confidence_vehicle=cv, confidence_plate=cp)
        pipeline = ALPRPipeline(models, options_from_args(args, snapshots_dir=None), th,
                                record_observations=True)
        for stem in videos:
            path = os.path.join(args.obs_dir, f"{stem}__cv{cv}_cp{cp}.json")
            if os.path.exists(path) and not args.overwrite:
                logging.info(f"[record] ya existe, se reutiliza: {path}")
                continue
            res = pipeline.run_video(find_video(args.videos_dir, stem))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res.observations, f)
            logging.info(f"[record] {path}")


# ------------------------------------------------------------------------------
# Paso 2: barrido offline
# ------------------------------------------------------------------------------
def load_observations(obs_dir):
    """-> {(cv, cp): {video: obs}}"""
    groups = {}
    for path in sorted(glob.glob(os.path.join(obs_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            obs = json.load(f)
        d = obs["detection_thresholds"]
        key = (d["confidence_vehicle"], d["confidence_plate"])
        groups.setdefault(key, {})[os.path.splitext(obs["source"])[0]] = obs
    if not groups:
        raise SystemExit(f"No hay observaciones en {obs_dir}: corré primero el paso `record`")
    return groups


def iter_grid(grid):
    for name in grid:
        if name not in DECISION_FIELDS:
            raise SystemExit(f"{name!r} no es un umbral de decisión barrible offline: {DECISION_FIELDS}")
    names = list(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        yield dict(zip(names, values))


def score_row(row, objective, min_precision):
    """Mayor es mejor. En `recall_at_precision`, lo que no llega a la precisión
    mínima queda siempre por debajo de lo que sí llega."""
    if objective == "f1":
        return (row["f1"], row["precision"])
    ok = row["precision"] >= min_precision
    return (1 if ok else 0, row["recall"] if ok else row["precision"], row["precision"])


def cmd_sweep(args):
    grid = DEFAULT_GRID
    if args.grid:
        with open(args.grid, "r", encoding="utf-8") as f:
            grid = json.load(f)
    gt = load_ground_truth(args.ground_truth)
    groups = load_observations(args.obs_dir)

    rows = []
    for (cv, cp), obs_by_video in groups.items():
        missing = [v for v in obs_by_video if v not in gt]
        if missing:
            raise SystemExit(f"Videos con observaciones pero sin GT: {missing}")
        gt_sub = {v: gt[v] for v in obs_by_video}
        strict = {o["strict_plate"] for o in obs_by_video.values()}
        if len(strict) != 1:
            raise SystemExit("Las observaciones mezclan modo estricto y permisivo")

        for combo in iter_grid(grid):
            th = Thresholds(confidence_vehicle=cv, confidence_plate=cp, **combo)
            preds = {v: replay_video(o, th, flush_unfinalized=not args.no_flush,
                                     dedup_seconds=args.dedup_seconds)
                     for v, o in obs_by_video.items()}
            m = evaluate_predictions(gt_sub, preds, args.tolerance_frames)
            rows.append({**th.to_dict(),
                         **{k: m[k] for k in ("n_gt", "n_pred", "tp", "wrong", "spurious",
                                              "duplicate", "fn", "precision", "recall", "f1")}})

    rows.sort(key=lambda r: score_row(r, args.objective, args.min_precision), reverse=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sweep_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    best = rows[0]
    best_th = Thresholds.from_dict({k: best[k] for k in Thresholds().to_dict()})
    reached = args.objective == "f1" or best["precision"] >= args.min_precision
    out = {
        "thresholds": best_th.to_dict(),
        "calibrated_on": sorted({v for g in groups.values() for v in g}),
        "objective": args.objective, "min_precision": args.min_precision,
        "objective_reached": reached,
        "metrics_on_calibration_set": {k: best[k] for k in
                                       ("tp", "wrong", "spurious", "fn", "precision", "recall", "f1")},
        "n_combinations": len(rows),
        "note": "Métricas sobre el set de CALIBRACIÓN (optimistas). El número citable sale de "
                "alpr.evaluate.e2e sobre la partición test.",
    }
    with open(os.path.join(args.out_dir, "best_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{len(rows)} combinaciones evaluadas sobre {len(out['calibrated_on'])} videos")
    print("Mejor:", json.dumps(best_th.to_dict()))
    print("  en calib:", out["metrics_on_calibration_set"])
    if not reached:
        print(f"AVISO: ninguna combinación alcanza precisión >= {args.min_precision}; "
              "se muestra la de mayor precisión.")
    n_gt = best["n_gt"]
    if n_gt < 50:
        print(f"AVISO: solo {n_gt} vehículos de GT en calibración; con tan pocos datos la grilla "
              "se sobreajusta. Desconfiá de diferencias chicas entre combinaciones.")


def main(argv=None):
    from ..inference import add_pipeline_args, add_threshold_args
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="Corre los modelos y graba las lecturas")
    r.add_argument("--splits", default=os.path.join("splits", "videos.json"))
    r.add_argument("--split", default="calib")
    r.add_argument("--videos-dir", default=os.path.join("data", "videos"))
    r.add_argument("--obs-dir", default=os.path.join("outputs", "calibration", "obs"))
    r.add_argument("--confidence-vehicle-grid", type=float, nargs="+", default=None)
    r.add_argument("--confidence-plate-grid", type=float, nargs="+", default=None)
    r.add_argument("--overwrite", action="store_true")
    add_pipeline_args(r)
    add_threshold_args(r)
    r.set_defaults(func=cmd_record)

    s = sub.add_parser("sweep", help="Barre umbrales de decisión offline contra el GT")
    s.add_argument("--obs-dir", default=os.path.join("outputs", "calibration", "obs"))
    s.add_argument("--ground-truth", default=os.path.join("data", "ground_truth.csv"))
    s.add_argument("--out-dir", default=os.path.join("outputs", "calibration"))
    s.add_argument("--grid", default=None, help="JSON {umbral: [valores]} (por defecto DEFAULT_GRID)")
    s.add_argument("--objective", choices=("f1", "recall_at_precision"), default="recall_at_precision")
    s.add_argument("--min-precision", type=float, default=0.95)
    s.add_argument("--tolerance-frames", type=int, default=0)
    s.add_argument("--dedup-seconds", type=float, default=30.0)
    s.add_argument("--no-flush", action="store_true", help="Sin recuperación de tracks sin consenso")
    s.set_defaults(func=cmd_sweep)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)
    args.func(args)


if __name__ == "__main__":
    main()
