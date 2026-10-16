"""Conversión CCPD -> COCO (una sola categoría: "plate", id 0).

El nombre de archivo de CCPD codifica la anotación, separado por '-':
  <área>-<inclinación>-<bbox>-<4 vértices>-<índices de caracteres>-<brillo>-<borrosidad>.jpg
  bbox = "x1&y1_x2&y2" (esquina superior izquierda e inferior derecha), en píxeles.

    python -m alpr.data.ccpd --images-dir data/CCPD2019 --out data/ccpd_coco.json

Luego partir con `python -m alpr.data.splits coco ...`.
NOTA: CCPD son patentes chinas. Sirve para entrenar el DETECTOR (la forma de
una chapa), no para calibrar ni evaluar sobre patentes argentinas: eso exige un
set propio etiquetado.
"""
import argparse
import json
import os

IMG_EXTS = (".jpg", ".jpeg", ".png")


def to_posix(path):
    """Rutas del JSON COCO siempre con '/': en Windows os.path.relpath devuelve
    barras invertidas, y eso rompe los --group-regex de alpr.data.splits y hace
    que los datasets no sean portables entre sistemas."""
    return str(path).replace("\\", "/")


def parse_ccpd_bbox(filename):
    """Devuelve (x1, y1, x2, y2) desde el nombre de archivo CCPD, o None si no parsea."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    try:
        (x1, y1), (x2, y2) = [tuple(int(v) for v in corner.split("&"))
                              for corner in parts[2].split("_")]
    except ValueError:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size   # (w, h), lee solo el header
    except ImportError:
        import cv2
        img = cv2.imread(path)
        return (img.shape[1], img.shape[0]) if img is not None else None


def build_coco(images_dir, rel_to=None):
    rel_to = rel_to or images_dir
    images, annotations, skipped = [], [], 0
    next_ann = 1
    for root, _, files in os.walk(images_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            box = parse_ccpd_bbox(fn)
            full = os.path.join(root, fn)
            size = _image_size(full) if box else None
            if box is None or size is None:
                skipped += 1
                continue
            w, h = size
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": to_posix(os.path.relpath(full, rel_to)),
                           "width": w, "height": h})
            annotations.append({"id": next_ann, "image_id": image_id, "category_id": 0,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            next_ann += 1
    return {"info": {"description": "CCPD -> COCO"},
            "categories": [{"id": 0, "name": "plate"}],
            "images": images, "annotations": annotations}, skipped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    coco, skipped = build_coco(args.images_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"{len(coco['images'])} imágenes convertidas, {skipped} omitidas -> {args.out}")


if __name__ == "__main__":
    main()
