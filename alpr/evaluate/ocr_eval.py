"""EVALUACIÓN / calibración del OCR sobre recortes de patente etiquetados.

    python -m alpr.evaluate.ocr_eval --labels data/ocr_calib/labels.csv \
        --crops-dir data/ocr_calib --strict-plate --attempts 1 2 3 4

labels.csv:  image,plate,vehicle_class      (vehicle_class opcional: car|motorcycle|...)

Corre el OCR adaptativo con 1, 2, 3 y 4 intentos máximos sobre los MISMOS
recortes: es el A/B de `--ocr-max-attempts` (cuánto rescatan las variantes
CLAHE / unsharp / Sauvola reforzada y cuánto cuestan). No carga los detectores.
Usá el CSV de calibración para elegir parámetros y el de test una sola vez.
"""
import argparse
import csv
import json
import logging
import os
import time

import cv2

from ..inference import add_threshold_args, thresholds_from_args
from ..ocr import PlateOCR
from .metrics import ocr_metrics


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"image", "plate"} <= set(rows[0]):
        raise ValueError("labels.csv necesita las columnas image,plate[,vehicle_class]")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels", required=True)
    p.add_argument("--crops-dir", required=True)
    p.add_argument("--attempts", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--strict-plate", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default=None, help="JSON de salida (opcional)")
    add_threshold_args(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', force=True)

    th = thresholds_from_args(args)
    rows = load_labels(args.labels)

    from ..models import load_ocr_reader
    reader = load_ocr_reader(gpu=not args.cpu)

    crops = []
    for r in rows:
        img = cv2.imread(os.path.join(args.crops_dir, r["image"]))
        if img is None:
            raise FileNotFoundError(os.path.join(args.crops_dir, r["image"]))
        crops.append((img, r["plate"], (r.get("vehicle_class") or "default").strip() or "default"))

    results = {}
    for n_attempts in args.attempts:
        ocr = PlateOCR(reader, th, args.strict_plate, dual_ocr=n_attempts > 1,
                       max_attempts=n_attempts)
        pairs, t0 = [], time.time()
        for img, gt_plate, vclass in crops:
            out = ocr.read(img, vclass)
            pairs.append((gt_plate, out[0] if out else None))
        m = ocr_metrics(pairs)
        m["seconds"] = round(time.time() - t0, 2)
        m["attempts_total"] = ocr.stats["ocr_attempts_total"]
        m["selected_by_variant"] = dict(ocr.stats["ocr_selected_by_variant"])
        results[str(n_attempts)] = m
        print(f"intentos<={n_attempts}: exact={m['exact_match']:.3f} lectura={m['read_rate']:.3f} "
              f"precisión={m['precision']:.3f} CER={m['cer_on_read']} "
              f"({m['attempts_total']} inferencias, {m['seconds']}s)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"thresholds": th.to_dict(), "strict_plate": args.strict_plate,
                       "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
