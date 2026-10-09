"""Particiones train / calib / test SIN fuga de datos.

Regla de oro: se parte por GRUPO (video, vehículo, secuencia), nunca por frame.
Frames consecutivos del mismo auto son casi idénticos; si uno cae en train y el
otro en test, las métricas de test quedan infladas.

  Videos (calibración / evaluación end-to-end):
      python -m alpr.data.splits videos --videos-dir data/videos \
          --ground-truth data/ground_truth.csv --split calib=0.5 --split test=0.5 \
          --out splits/videos.json

  Imágenes COCO (entrenamiento del detector de patentes):
      python -m alpr.data.splits coco --annotations data/ccpd_coco.json \
          --split train=0.8 --split val=0.1 --split test=0.1 --out-dir data/detector
      # Si varias imágenes pertenecen a una misma secuencia/vehículo, indicalo con
      # --group-regex (el grupo 1 de la regex sobre file_name define el grupo).
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def split_groups(groups, ratios, seed=42):
    """Reparte `groups` (iterable de claves hashables) según `ratios`
    ({nombre: proporción}). Determinista dada la semilla. Cada partición con
    ratio > 0 recibe al menos 1 grupo; si no alcanzan los grupos, falla en
    lugar de devolver una partición vacía en silencio."""
    groups = sorted(set(groups))
    names = list(ratios)
    if any(r < 0 for r in ratios.values()) or sum(ratios.values()) <= 0:
        raise ValueError("Las proporciones deben ser >= 0 y sumar > 0")
    positive = [n for n in names if ratios[n] > 0]
    if len(groups) < len(positive):
        raise ValueError(f"Hay {len(groups)} grupos y {len(positive)} particiones no vacías: "
                         "faltan datos para partir sin dejar una partición vacía")

    rng = random.Random(seed)
    rng.shuffle(groups)

    total = float(sum(ratios.values()))
    counts = {n: int(len(groups) * ratios[n] / total) for n in names}
    for n in positive:                      # mínimo 1 por partición no vacía
        counts[n] = max(1, counts[n])
    # Ajusta el sobrante / faltante sobre la partición más grande.
    biggest = max(positive, key=lambda n: ratios[n])
    counts[biggest] += len(groups) - sum(counts.values())
    if counts[biggest] < 1:
        raise ValueError("No alcanzan los grupos para respetar las proporciones pedidas")

    out, i = {}, 0
    for n in names:
        out[n] = sorted(groups[i:i + counts[n]])
        i += counts[n]
    return out


def assert_disjoint(splits):
    """Falla si algún elemento aparece en más de una partición."""
    seen = {}
    for name, items in splits.items():
        for it in items:
            if it in seen:
                raise ValueError(f"FUGA DE DATOS: {it!r} está en {seen[it]!r} y en {name!r}")
            seen[it] = name


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert_disjoint(data["splits"])
    return data


def load_split(path, name):
    data = load_manifest(path)
    if name not in data["splits"]:
        raise KeyError(f"La partición {name!r} no existe en {path}: {list(data['splits'])}")
    return data["splits"][name]


def _parse_ratios(items):
    ratios = {}
    for it in items:
        name, _, val = it.partition("=")
        if not name or not val:
            raise ValueError(f"--split espera nombre=proporción, recibí {it!r}")
        ratios[name] = float(val)
    return ratios


def video_names_from_ground_truth(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sorted({row["video"] for row in csv.DictReader(f)})


def cmd_videos(args):
    ratios = _parse_ratios(args.split)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.videos_dir)
                   if f.lower().endswith(VIDEO_EXTS))
    if args.ground_truth:
        labeled = set(video_names_from_ground_truth(args.ground_truth))
        missing = sorted(labeled - set(stems))
        if missing:
            print(f"AVISO: hay GT para videos que no están en {args.videos_dir}: {missing}",
                  file=sys.stderr)
        stems = [s for s in stems if s in labeled]
    splits = split_groups(stems, ratios, args.seed)
    assert_disjoint(splits)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"kind": "videos", "seed": args.seed, "ratios": ratios, "splits": splits},
                  f, indent=2, ensure_ascii=False)
    for n, items in splits.items():
        print(f"{n}: {len(items)} videos")


def cmd_coco(args):
    ratios = _parse_ratios(args.split)
    with open(args.annotations, "r", encoding="utf-8") as f:
        coco = json.load(f)
    rx = re.compile(args.group_regex) if args.group_regex else None

    group_of = {}
    for img in coco["images"]:
        if rx:
            m = rx.search(img["file_name"])
            group_of[img["id"]] = m.group(1) if m and m.groups() else img["file_name"]
        else:
            group_of[img["id"]] = str(img["id"])    # cada imagen es su propio grupo
    ids_by_group = defaultdict(list)
    for iid, g in group_of.items():
        ids_by_group[g].append(iid)

    groups_split = split_groups(ids_by_group.keys(), ratios, args.seed)
    assert_disjoint(groups_split)

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    images_by_id = {i["id"]: i for i in coco["images"]}

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for name, groups in groups_split.items():
        ids = sorted(iid for g in groups for iid in ids_by_group[g])
        sub = {
            "info": coco.get("info", {}),
            "categories": coco["categories"],
            "images": [images_by_id[i] for i in ids],
            "annotations": [a for i in ids for a in anns_by_image.get(i, [])],
        }
        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sub, f)
        summary[name] = {"images": len(sub["images"]), "annotations": len(sub["annotations"]),
                         "groups": len(groups), "file": path}
        print(f"{name}: {len(sub['images'])} imágenes, {len(sub['annotations'])} anotaciones -> {path}")
    with open(os.path.join(args.out_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"kind": "coco", "seed": args.seed, "ratios": ratios, "summary": summary},
                  f, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("videos")
    v.add_argument("--videos-dir", required=True)
    v.add_argument("--ground-truth", default=None, help="CSV de GT: solo se usan los videos etiquetados")
    v.add_argument("--split", action="append", required=True, metavar="NOMBRE=PROPORCIÓN")
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--out", default=os.path.join("splits", "videos.json"))
    v.set_defaults(func=cmd_videos)

    c = sub.add_parser("coco")
    c.add_argument("--annotations", required=True)
    c.add_argument("--split", action="append", required=True, metavar="NOMBRE=PROPORCIÓN")
    c.add_argument("--group-regex", default=None)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out-dir", required=True)
    c.set_defaults(func=cmd_coco)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
