"""Métricas de evaluación (puras: sin modelos ni GPU).

Verdad-terreno end-to-end (CSV, una fila por VEHÍCULO que pasa por la zona de lectura):

    video,plate,first_frame,last_frame,vehicle_type
    trafico01,AB123CD,120,310,car

`video` es el nombre del archivo SIN extensión. [first_frame, last_frame] es el
tramo en que ese vehículo es visible y legible; una detección cuenta para ese
vehículo si su `frame_idx` cae dentro del tramo (± tolerancia).
`vehicle_type` es opcional.
"""
import csv
from collections import defaultdict
from dataclasses import dataclass


def normalize_plate(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


@dataclass(frozen=True)
class GTEntry:
    video: str
    plate: str
    first_frame: int
    last_frame: int
    vehicle_type: str = ""


def load_ground_truth(path):
    """CSV -> {video: [GTEntry]}. Valida lo que un error de etiquetado rompería."""
    by_video = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video", "plate", "first_frame", "last_frame"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Al CSV de GT le faltan columnas: {sorted(missing)}")
        for n, row in enumerate(reader, start=2):
            plate = normalize_plate(row["plate"])
            first, last = int(row["first_frame"]), int(row["last_frame"])
            if not plate:
                raise ValueError(f"{path}:{n}: patente vacía")
            if first < 0 or last < first:
                raise ValueError(f"{path}:{n}: tramo inválido [{first}, {last}]")
            by_video[row["video"].strip()].append(
                GTEntry(row["video"].strip(), plate, first, last,
                        (row.get("vehicle_type") or "").strip()))
    return dict(by_video)


def levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ratios(tp, wrong, spurious, n_gt):
    n_out = tp + wrong + spurious
    precision = tp / n_out if n_out else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def match_video(gts, preds, tolerance_frames=0):
    """Asocia detecciones con vehículos de GT de UN video.

    Cada detección termina en exactamente una categoría:
      tp        patente correcta sobre un vehículo todavía no leído bien
      duplicate patente correcta sobre un vehículo que ya tenía su TP (no penaliza)
      wrong     cae dentro del tramo de un vehículo pero con otra patente
      spurious  no cae en el tramo de ningún vehículo (falso positivo puro)
    Cada vehículo de GT sin TP es un falso negativo.
    """
    tol = int(tolerance_frames)

    def inside(g, f):
        return g.first_frame - tol <= f <= g.last_frame + tol

    order = sorted(range(len(preds)), key=lambda i: preds[i]["frame_idx"])
    status, tp_gt = {}, set()

    # Pasada 1: lecturas correctas (se prioriza que cada vehículo tenga su TP).
    for i in order:
        f, plate = preds[i]["frame_idx"], normalize_plate(preds[i]["patente_validada"])
        cands = [j for j, g in enumerate(gts) if g.plate == plate and inside(g, f)]
        if not cands:
            continue
        free = [j for j in cands if j not in tp_gt]
        if free:
            j = min(free, key=lambda j: abs((gts[j].first_frame + gts[j].last_frame) / 2 - f))
            tp_gt.add(j)
            status[i] = "tp"
        else:
            status[i] = "duplicate"

    # Pasada 2: lo que queda es lectura errónea o falso positivo puro.
    wrong_dist = {}   # j -> menor distancia de edición entre sus lecturas erróneas
    for i in order:
        if i in status:
            continue
        f, plate = preds[i]["frame_idx"], normalize_plate(preds[i]["patente_validada"])
        cands = [j for j, g in enumerate(gts) if inside(g, f)]
        if cands:
            status[i] = "wrong"
            j = min(cands, key=lambda j: levenshtein(plate, gts[j].plate))
            d = levenshtein(plate, gts[j].plate)
            wrong_dist[j] = min(d, wrong_dist.get(j, d))
        else:
            status[i] = "spurious"

    counts = defaultdict(int)
    for st in status.values():
        counts[st] += 1
    read_gts = tp_gt | set(wrong_dist)
    edit = sum(0 if j in tp_gt else wrong_dist[j] for j in read_gts)
    chars = sum(len(gts[j].plate) for j in read_gts)
    return {
        "n_gt": len(gts), "n_pred": len(preds),
        "tp": counts["tp"], "duplicate": counts["duplicate"],
        "wrong": counts["wrong"], "spurious": counts["spurious"],
        "fn": len(gts) - len(tp_gt),
        "cer_edit": edit, "cer_chars": chars,
    }


def evaluate_predictions(gt_by_video, preds_by_video, tolerance_frames=0, finalizations=None):
    """Métricas globales y por video. `finalizations`: conjunto de valores de
    `finalization` a conservar (p. ej. {"consensus"}); None = todas."""
    per_video, total = {}, defaultdict(int)
    for video, gts in gt_by_video.items():
        preds = preds_by_video.get(video, [])
        if finalizations is not None:
            preds = [p for p in preds if p.get("finalization") in finalizations]
        r = match_video(gts, preds, tolerance_frames)
        per_video[video] = _finish(r)
        for k, v in r.items():
            total[k] += v
    out = _finish(dict(total))
    out["per_video"] = per_video
    return out


def _finish(r):
    precision, recall, f1 = _ratios(r["tp"], r["wrong"], r["spurious"], r["n_gt"])
    r = dict(r)
    r.update(precision=precision, recall=recall, f1=f1,
             cer_on_read=(r["cer_edit"] / r["cer_chars"]) if r["cer_chars"] else None)
    return r


def ocr_metrics(pairs):
    """pairs: [(plate_gt, plate_pred | None)] sobre crops individuales.
      exact_match   aciertos / TODOS los crops (None cuenta como fallo)
      read_rate     crops con alguna lectura / todos
      precision     aciertos / crops con lectura
      cer_on_read   distancia de edición / caracteres, solo sobre lo leído"""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    read = [(normalize_plate(g), normalize_plate(p)) for g, p in pairs if p]
    hits = sum(1 for g, p in read if g == p)
    chars = sum(len(g) for g, _ in read)
    edit = sum(levenshtein(p, g) for g, p in read)
    return {
        "n": n, "n_read": len(read), "n_exact": hits,
        "exact_match": hits / n,
        "read_rate": len(read) / n,
        "precision": hits / len(read) if read else 0.0,
        "cer_on_read": edit / chars if chars else None,
    }
