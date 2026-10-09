"""EVALUACIÓN del detector de patentes (RT-DETR) sobre una partición COCO.

    python -m alpr.evaluate.detector_eval --model outputs/rtdetr-ccpd/final_model \
        --annotations data/detector/test.json --images-dir data/CCPD2019

Reporta AP@0.5, AP@[.5:.95], y precisión/recall/IoU medio a varios umbrales de
score (de ahí sale el `confidence_plate` razonable para el pipeline). Usá `val`
para elegir el umbral y `test` una sola vez para el número final.
"""
import argparse
import json
import os

import cv2

from .detection_metrics import average_precision, coco_style_ap, precision_recall_at


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--score-floor", type=float, default=0.01,
                   help="Piso al post-procesar (bajo, para poder barrer umbrales después)")
    p.add_argument("--score-thresholds", type=float, nargs="+",
                   default=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    import torch
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model).to(device).eval()

    with open(args.annotations, "r", encoding="utf-8") as f:
        coco = json.load(f)
    gts_by_image = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gts_by_image.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    images = coco["images"][:args.max_images] if args.max_images else coco["images"]

    preds, gts = [], []
    for start in range(0, len(images), args.batch):
        chunk = images[start:start + args.batch]
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(args.images_dir, i["file_name"])),
                             cv2.COLOR_BGR2RGB) for i in chunk]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        sizes = torch.tensor([im.shape[:2] for im in imgs]).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs, target_sizes=sizes, threshold=args.score_floor)
            for info, res in zip(chunk, results):
                boxes = res["boxes"].float().cpu().numpy()
                scores = res["scores"].float().cpu().numpy()
                preds.append([(b.tolist(), float(s)) for b, s in zip(boxes, scores)])
                gts.append(gts_by_image.get(info["id"], []))

    report = {
        "n_images": len(images), "n_gt": sum(len(g) for g in gts),
        "AP50": average_precision(preds, gts, 0.5),
        "AP75": average_precision(preds, gts, 0.75),
        "AP50_95": coco_style_ap(preds, gts),
        "by_score_threshold": [precision_recall_at(preds, gts, t, 0.5) for t in args.score_thresholds],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
