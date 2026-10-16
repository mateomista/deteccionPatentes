"""Une las detecciones del pipeline con las clasificaciones del consumidor.

    python -m alpr.tools.join_results --detections outputs/trafico01.json \
        --classifications outputs/clasificaciones_vehiculos.jsonl --video trafico01 \
        --out outputs/trafico01_unido.json

La clave es (video, track_id): `id_vehiculo` de la detección contra `track_id` + `video` de la
clasificación (el video viaja en el nombre del recorte, ver alpr/contracts.py).

Solo usa la biblioteca estándar: se puede copiar suelto junto a los scripts de standalone/.
"""
import argparse
import json
import os
import re
import sys

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
OK_STATES = {"ok", "below_threshold", "mock"}


def sanitize_video_stem(stem):
    # Misma regla que alpr/contracts.py (una prueba verifica que no diverjan).
    return _UNSAFE.sub("-", str(stem)).strip("-._") or "video"


def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: se esperaba una lista de detecciones")
    return data


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"AVISO: {path}:{n} no es JSON válido, se ignora", file=sys.stderr)
    return rows


def join_results(detections, classifications, video, accept_legacy=False):
    """Devuelve (unidas, detecciones_sin_clasificar, clasificaciones_sin_detección).

    `accept_legacy=True` acepta también clasificaciones sin video (nombre antiguo
    `vehiculo_<id>.jpg`), útil solo si hay un único video: de lo contrario son ambiguas."""
    best = {}
    for row in classifications:
        if row.get("estado") not in OK_STATES:
            continue
        vid = row.get("video")
        if vid is None:
            if not accept_legacy:
                continue
        elif vid != video:
            continue
        best[int(row["track_id"])] = row        # la última clasificación válida gana

    joined, no_class, used = [], [], set()
    for det in detections:
        tid = det.get("id_vehiculo")
        row = best.get(int(tid)) if tid is not None else None
        rec = dict(det)
        if row is None:
            rec["clasificacion"] = None
            no_class.append(det)
        else:
            used.add(int(tid))
            rec["clasificacion"] = {k: row.get(k) for k in
                                    ("etiqueta", "marca", "modelo", "confianza", "estado")}
        joined.append(rec)
    orphans = [row for tid, row in best.items() if tid not in used]
    return joined, no_class, orphans


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--detections", required=True, help="JSON de detecciones del pipeline")
    p.add_argument("--classifications", required=True, help="JSONL del clasificador")
    p.add_argument("--video", default=None,
                   help="Nombre del video (sin extensión). Por defecto, el nombre del archivo de detecciones")
    p.add_argument("--accept-legacy", action="store_true",
                   help="Acepta clasificaciones sin video (nombre antiguo vehiculo_<id>.jpg)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    video = sanitize_video_stem(args.video or os.path.splitext(os.path.basename(args.detections))[0])
    joined, no_class, orphans = join_results(
        load_json_list(args.detections), load_jsonl(args.classifications), video, args.accept_legacy)

    print(f"video={video}: {len(joined)} detecciones | clasificadas={len(joined) - len(no_class)} | "
          f"sin clasificar={len(no_class)} | clasificaciones sin detección={len(orphans)}")
    for d in no_class:
        print(f"  sin clasificar: ID {d.get('id_vehiculo')} {d.get('patente_validada')}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(joined, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
