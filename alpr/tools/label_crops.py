"""Arma el conjunto de recortes de patente ETIQUETADOS para `alpr.evaluate.ocr_eval`.

Parte de la carpeta que escribe `--export-plate-crops` (indice.json) y sigue dos pasos:

  1) Plantilla: una fila por vehículo que tiene recortes para el OCR. Completás la columna `plate` con
     la patente REAL, mirando el video o el recorte de ejemplo, y dejás vacía la de los vehículos que
     no se leen.

        python -m alpr.tools.label_crops template --crops-dir outputs/crops_patente \
            --out outputs/crops_patente/vehiculos.csv

  2) Armado: cada recorte hereda la patente de su vehículo y se copia, con su etiqueta, a una carpeta
     que `ocr_eval` ya sabe leer.

        python -m alpr.tools.label_crops build --crops-dir outputs/crops_patente \
            --vehicles outputs/crops_patente/vehiculos.csv --out-dir data/ocr_calib

        python -m alpr.evaluate.ocr_eval --labels data/ocr_calib/labels.csv \
            --crops-dir data/ocr_calib --strict-plate

La plantilla NO muestra lo que leyó el OCR (salvo con --show-reads): una lectura equivocada con aspecto
de patente condiciona al que etiqueta. Por eso el recorte de ejemplo de cada vehículo se copia a una carpeta
`muestras/` con un nombre neutro (los archivos originales llevan la lectura del OCR en el nombre). Si un vehículo tiene recortes que no se pueden leer (la patente
sale cortada, por ejemplo), anotá sus cuadros en `excluir_frames` (separados por coma) y no se copian.
`build` se puede repetir con otros videos: suma a labels.csv y reemplaza las filas de la misma imagen.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict

from ..evaluate.metrics import normalize_plate
from ..plate_text import matches_official_format

OCR_KINDS = ("ocr_ok", "ocr_sin_lectura")
VEHICLE_FIELDS = ["id_vehiculo", "tipo_vehiculo", "recortes_ocr", "cuadros", "imagen_ejemplo",
                  "plate", "excluir_frames"]


def load_index(crops_dir):
    path = os.path.join(crops_dir, "indice.json")
    if not os.path.exists(path):
        raise SystemExit(f"No existe {path}: correr alpr.inference con --export-plate-crops")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sample_crop(items):
    """El recorte más grande: es el que más probablemente se pueda leer a ojo."""
    return max(items, key=lambda it: it["ancho"] * it["alto"])


def neutral_sample_name(item):
    """Nombre del recorte de ejemplo SIN la lectura del OCR (el archivo original la lleva en el nombre)."""
    return f"ID{item['id_vehiculo']}_F{item['frame']:06d}.jpg"


def vehicle_rows(index, show_reads=False):
    """Una fila por vehículo con al menos un recorte para el OCR. `imagen_ejemplo` es el nombre
    neutro (ver `neutral_sample_name`) de su recorte más grande."""
    by_vehicle = defaultdict(list)
    for it in index["recortes"]:
        if it["tipo"] in OCR_KINDS:
            by_vehicle[it["id_vehiculo"]].append(it)
    rows = []
    for tid in sorted(by_vehicle):
        items = sorted(by_vehicle[tid], key=lambda i: i["frame"])
        row = {"id_vehiculo": tid, "tipo_vehiculo": items[0].get("tipo_vehiculo") or "",
               "recortes_ocr": len(items), "cuadros": f"{items[0]['frame']}-{items[-1]['frame']}",
               "imagen_ejemplo": "muestras/" + neutral_sample_name(_sample_crop(items)),
               "plate": "", "excluir_frames": ""}
        if show_reads:
            reads = sorted({it["texto"] for it in items if it.get("texto")})
            row["lecturas_ocr"] = " | ".join(reads)
        rows.append(row)
    return rows


def copy_samples(index, crops_dir, samples_dir):
    """Copia el recorte más grande de cada vehículo a `samples_dir` con un nombre neutro."""
    by_vehicle = defaultdict(list)
    for it in index["recortes"]:
        if it["tipo"] in OCR_KINDS:
            by_vehicle[it["id_vehiculo"]].append(it)
    os.makedirs(samples_dir, exist_ok=True)
    for items in by_vehicle.values():
        best = _sample_crop(items)
        shutil.copyfile(os.path.join(crops_dir, best["archivo"]),
                        os.path.join(samples_dir, neutral_sample_name(best)))


def _read_csv(path):
    """Lee un CSV con coma o punto y coma (Excel en español guarda con punto y coma) y con o sin BOM."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    header = text.splitlines()[0] if text else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def parse_frames(text):
    out = set()
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                raise SystemExit(f"excluir_frames tiene un valor que no es un número de cuadro: {part!r}")
    return out


def build_labels(index, vehicle_rows_, crops_dir, include_focus_mask=False):
    """Decide qué recortes se etiquetan, SIN tocar el disco. Devuelve (filas de labels.csv,
    copias [(origen, nombre)], avisos, resumen)."""
    kinds = OCR_KINDS + (("focus_mask",) if include_focus_mask else ())
    by_vehicle = defaultdict(list)
    for it in index["recortes"]:
        if it["tipo"] in kinds:
            by_vehicle[it["id_vehiculo"]].append(it)
    video = index.get("video", "video")
    rows, copies, warnings, summary = [], [], [], []
    for r in vehicle_rows_:
        try:
            tid = int(r["id_vehiculo"])
        except (KeyError, ValueError, TypeError):
            warnings.append(f"fila sin id_vehiculo válido: {r}")
            continue
        plate = normalize_plate(r.get("plate") or "")
        if not plate:
            continue
        if not matches_official_format(plate):
            warnings.append(f"#{tid}: {plate!r} no tiene formato de patente argentina "
                            f"(LLLNNN, LLNNNLL o LNNNLLL); revisá que esté bien escrita")
        items = by_vehicle.get(tid, [])
        if not items:
            warnings.append(f"#{tid}: tiene patente pero no tiene recortes para el OCR")
            continue
        skip = parse_frames(r.get("excluir_frames"))
        kept = 0
        for it in items:
            if it["frame"] in skip:
                continue
            name = f"{video}__F{it['frame']:06d}_ID{tid}.jpg"
            src = os.path.join(crops_dir, it["archivo"])
            if not os.path.exists(src):
                warnings.append(f"#{tid}: falta el archivo {it['archivo']}")
                continue
            copies.append((src, name))
            rows.append({"image": name, "plate": plate, "vehicle_class": it.get("tipo_vehiculo") or ""})
            kept += 1
        summary.append((tid, plate, kept, len(items) - kept))
    return rows, copies, warnings, summary


def merge_labels(path, new_rows):
    """Suma `new_rows` a labels.csv (si existe); la misma imagen se reemplaza."""
    merged = {}
    if os.path.exists(path):
        for r in _read_csv(path):
            merged[r["image"]] = {"image": r["image"], "plate": r["plate"],
                                  "vehicle_class": r.get("vehicle_class", "")}
    for r in new_rows:
        merged[r["image"]] = r
    with open(path, "w", newline="", encoding="utf-8") as f:   # sin BOM: ocr_eval lo lee con utf-8
        w = csv.DictWriter(f, fieldnames=["image", "plate", "vehicle_class"])
        w.writeheader()
        w.writerows(merged[k] for k in sorted(merged))
    return len(merged)


def cmd_template(args):
    index = load_index(args.crops_dir)
    rows = vehicle_rows(index, args.show_reads)
    if not rows:
        raise SystemExit("No hay recortes para el OCR en esa carpeta")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} ya existe (puede tener patentes cargadas): usar --overwrite para pisarlo")
    copy_samples(index, args.crops_dir, os.path.join(os.path.dirname(os.path.abspath(args.out)), "muestras"))
    fields = VEHICLE_FIELDS + (["lecturas_ocr"] if args.show_reads else [])
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} vehículos con recortes para el OCR -> {args.out}")
    print("Completá la columna `plate` con la patente REAL (vacía si no se lee) y corré `build`.")


def cmd_build(args):
    index = load_index(args.crops_dir)
    vehicles = _read_csv(args.vehicles)
    rows, copies, warnings, summary = build_labels(index, vehicles, args.crops_dir,
                                                   args.include_focus_mask)
    for w in warnings:
        print(f"[AVISO] {w}", file=sys.stderr)
    missing = [w for w in warnings if "falta el archivo" in w]
    if missing and not args.allow_missing:
        raise SystemExit(f"Faltan {len(missing)} archivos de recortes: no se escribió nada, porque "
                         f"etiquetar solo una parte de un vehículo da un conjunto sesgado. Revisá que "
                         f"{args.crops_dir} sea la carpeta de la misma corrida que el índice "
                         f"(o usá --allow-missing).")
    if not rows:
        raise SystemExit("No se etiquetó ningún recorte: completá la columna `plate` en el CSV de vehículos")
    os.makedirs(args.out_dir, exist_ok=True)
    for src, name in copies:
        shutil.copyfile(src, os.path.join(args.out_dir, name))
    total = merge_labels(os.path.join(args.out_dir, "labels.csv"), rows)
    for tid, plate, kept, skipped in summary:
        print(f"  #{tid}: {plate}  {kept} recortes" + (f" ({skipped} excluidos)" if skipped else ""))
    print(f"{len(rows)} recortes etiquetados; labels.csv tiene {total} filas -> "
          f"{os.path.join(args.out_dir, 'labels.csv')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="Genera el CSV de vehículos para completar")
    t.add_argument("--crops-dir", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--show-reads", action="store_true",
                   help="Agrega lo que leyó el OCR (puede condicionar al que etiqueta)")
    t.add_argument("--overwrite", action="store_true")
    t.set_defaults(func=cmd_template)
    b = sub.add_parser("build", help="Copia los recortes con su etiqueta y escribe labels.csv")
    b.add_argument("--crops-dir", required=True)
    b.add_argument("--vehicles", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--allow-missing", action="store_true",
                   help="Sigue aunque falten archivos de recortes (por defecto frena sin escribir nada)")
    b.add_argument("--include-focus-mask", action="store_true",
                   help="Incluye también los recortes que descartó el focus mask")
    b.set_defaults(func=cmd_build)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
