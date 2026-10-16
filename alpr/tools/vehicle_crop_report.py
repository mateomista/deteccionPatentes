"""Diagnóstico: ¿hay lecturas de patente en que el vehículo está COMPLETO dentro del cuadro?

    python -m alpr.tools.vehicle_crop_report --source data/videos/clip.mp4 \
        --plate-model outputs/final_model --strict-plate

El recorte de vehículo que se exporta sale del frame de la mejor lectura de la patente, y ese
momento suele ser el de mayor cercanía, cuando el auto más probablemente se sale del cuadro.
Este reporte corre el pipeline en modo de grabación y, por cada vehículo, muestra todas sus
lecturas: en cuáles la caja toca el borde del cuadro y en cuáles no. Sirve para decidir si vale
la pena elegir otro frame para el recorte, o si no hay mejor alternativa.
"""
import argparse
import json
import logging
import os
import sys


def touches_edge(bbox, frame_size, margin=5):
    """True si la caja [x1, y1, x2, y2] llega a menos de `margin` px de algún borde del cuadro."""
    x1, y1, x2, y2 = bbox
    w, h = frame_size
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def analyze_observations(obs, margin=5):
    """Una fila por vehículo con sus lecturas, separando las de caja completa de las cortadas."""
    if "frame_size" not in obs:
        raise ValueError("Las observaciones no traen frame_size: volvé a grabarlas con esta versión")
    frame_size = obs["frame_size"]
    rows = []
    for tid, track in obs["tracks"].items():
        votes = []
        for v in track["votes"]:
            votes.append({"frame": v["frame"], "text": v["text"], "conf": v["conf"], "bbox": v["bbox"],
                          "toca_borde": touches_edge(v["bbox"], frame_size, margin)})
        complete = [v for v in votes if not v["toca_borde"]]
        rows.append({
            "id_vehiculo": int(tid), "tipo_vehiculo": track["vehicle_class"],
            "lecturas": len(votes), "lecturas_completas": len(complete),
            "mejor_lectura": max(votes, key=lambda v: v["conf"]) if votes else None,
            "mejor_completa": max(complete, key=lambda v: v["conf"]) if complete else None,
            "votes": votes,
        })
    return sorted(rows, key=lambda r: r["id_vehiculo"])


def _fmt(v):
    if v is None:
        return "-"
    return f"F{v['frame']} {v['text']} conf={v['conf']:.2f} caja={v['bbox']}"


def format_report(rows, frame_size, margin):
    lines = [f"Cuadro {frame_size[0]}x{frame_size[1]}; 'toca el borde' = a menos de {margin} px de un borde.",
             ""]
    complete_vehicles = 0
    for r in rows:
        if r["mejor_completa"] is not None:
            complete_vehicles += 1
        lines.append(f"ID {r['id_vehiculo']} ({r['tipo_vehiculo']}): {r['lecturas']} lecturas, "
                     f"{r['lecturas_completas']} con el vehículo completo")
        if r["mejor_lectura"]:
            lines.append(f"    mejor lectura:         {_fmt(r['mejor_lectura'])}"
                         f"{'  <- TOCA EL BORDE' if r['mejor_lectura']['toca_borde'] else ''}")
            lines.append(f"    mejor con auto completo: {_fmt(r['mejor_completa'])}")
    lines += ["", f"Vehículos con al menos una lectura de caja completa: {complete_vehicles} de {len(rows)}"]
    return "\n".join(lines)


def main(argv=None):
    from ..inference import add_pipeline_args, add_threshold_args, options_from_args, thresholds_from_args
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--margin", type=int, default=5, help="px al borde para considerar que la caja lo toca")
    p.add_argument("--out", default=None, help="Guarda el detalle completo en JSON")
    add_pipeline_args(p)
    add_threshold_args(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s', force=True)

    from ..models import ModelBundle
    from ..pipeline import ALPRPipeline
    models = ModelBundle(args.plate_model, cudnn_benchmark=args.cudnn_benchmark)
    pipeline = ALPRPipeline(models, options_from_args(args, snapshots_dir=None),
                            thresholds_from_args(args), record_observations=True)
    obs = pipeline.run_video(args.source).observations
    rows = analyze_observations(obs, args.margin)
    print(format_report(rows, obs["frame_size"], args.margin))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
