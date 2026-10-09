"""Métricas del detector de patentes (una clase): IoU, AP@IoU y P/R a un umbral.

Implementación propia en numpy para no depender de pycocotools/torchmetrics.
Cajas en formato xyxy. `preds`: lista (por imagen) de [(box, score), ...];
`gts`: lista (por imagen) de [box, ...].
"""
import numpy as np


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _match(preds, gts, iou_thr):
    """Emparejamiento greedy por score. Devuelve (scores, is_tp, ious_de_tp, n_gt)."""
    records, ious, n_gt = [], [], 0
    for img_preds, img_gts in zip(preds, gts):
        n_gt += len(img_gts)
        used = set()
        for box, score in sorted(img_preds, key=lambda p: -p[1]):
            best, best_j = 0.0, -1
            for j, g in enumerate(img_gts):
                if j in used:
                    continue
                v = iou(box, g)
                if v > best:
                    best, best_j = v, j
            if best >= iou_thr:
                used.add(best_j)
                records.append((score, True))
                ious.append(best)
            else:
                records.append((score, False))
    return records, ious, n_gt


def average_precision(preds, gts, iou_thr=0.5):
    """AP con interpolación de todos los puntos (estilo VOC2010+)."""
    records, _, n_gt = _match(preds, gts, iou_thr)
    if n_gt == 0 or not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([1 if ok else 0 for _, ok in records])
    fp = np.cumsum([0 if ok else 1 for _, ok in records])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def precision_recall_at(preds, gts, score_thr, iou_thr=0.5):
    kept = [[(b, s) for b, s in img if s >= score_thr] for img in preds]
    records, ious, n_gt = _match(kept, gts, iou_thr)
    tp = sum(1 for _, ok in records if ok)
    n_pred = len(records)
    return {
        "score_threshold": score_thr, "iou_threshold": iou_thr,
        "precision": tp / n_pred if n_pred else 0.0,
        "recall": tp / n_gt if n_gt else 0.0,
        "mean_iou_tp": float(np.mean(ious)) if ious else None,
        "tp": tp, "fp": n_pred - tp, "fn": n_gt - tp,
    }


def coco_style_ap(preds, gts):
    """AP promediado en IoU 0.50:0.05:0.95 (equivalente a mAP de COCO con 1 clase)."""
    thrs = [0.5 + 0.05 * i for i in range(10)]
    return float(np.mean([average_precision(preds, gts, t) for t in thrs]))
