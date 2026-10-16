"""Compara dos corridas del pipeline sobre el MISMO video (p. ej. V5 vs modular).

    python -m alpr.tools.compare_runs --a outputs/v5.json --b outputs/modular.json \
        --label-a V5 --label-b modular

Lee los JSON de detecciones (lista de objetos con `patente_validada`, `frame_idx`,
`id_vehiculo`, ...) o el .jsonl incremental. Para comparar varios videos de una vez,
pasá varias rutas en --a y --b: se emparejan por orden.

Cada detección termina en una categoría:
  coincide     misma patente en ambos (se empareja por frame más cercano)
  lectura      mismo vehículo pero distinta patente: mismo `id_vehiculo`, o patentes a
               <= --max-edit caracteres y a <= --near-frames frames. NO es una
               coincidencia: es justo el tipo de diferencia que hay que mirar
  solo A / B   aparece en una sola corrida

Veredicto de PARIDAD (exit 0): ninguna detección solo-A/solo-B, ninguna diferencia de
lectura y desfase de frames <= --frame-tolerance. Cualquier otra cosa sale con exit 1.
La tolerancia por defecto (5 frames = un intervalo de OCR) cubre el no-determinismo
normal de la GPU; poné 0 para exigir identidad exacta de frame.
"""
import argparse
import json
import os
import sys

from ..evaluate.metrics import levenshtein, normalize_plate

COMPARED_FIELDS = ("frame_idx", "confianza_ocr_promedio", "cantidad_votos",
                   "finalization", "id_vehiculo", "tipo_vehiculo")


def load_detections(path):
    """Lee un JSON (lista) o un .jsonl de detecciones."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.strip()
    if not text:
        return []
    if path.lower().endswith(".jsonl"):
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path}: se esperaba una lista de detecciones")
    for d in data:
        if "patente_validada" not in d or "frame_idx" not in d:
            raise ValueError(f"{path}: detección sin patente_validada/frame_idx: {d}")
    return data


def _pair_greedy(items_a, items_b, cost):
    """Empareja por costo mínimo (greedy sobre todos los pares). Devuelve
    (pares, restantes_a, restantes_b)."""
    cands = sorted(((cost(a, b), i, j) for i, a in enumerate(items_a)
                    for j, b in enumerate(items_b) if cost(a, b) is not None),
                   key=lambda t: t[0])
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((items_a[i], items_b[j]))
    rest_a = [a for i, a in enumerate(items_a) if i not in used_a]
    rest_b = [b for j, b in enumerate(items_b) if j not in used_b]
    return pairs, rest_a, rest_b


def compare_detections(a, b, max_edit=2, near_frames=150):
    """Compara dos listas de detecciones. Devuelve un dict con las categorías."""
    # 1) misma patente: cada una con su par de frame más cercano.
    def same_plate_cost(x, y):
        if normalize_plate(x["patente_validada"]) != normalize_plate(y["patente_validada"]):
            return None
        return abs(int(x["frame_idx"]) - int(y["frame_idx"]))

    matched, rest_a, rest_b = _pair_greedy(a, b, same_plate_cost)

    # 2) mismo vehículo, otra lectura.
    def near_cost(x, y):
        edit = levenshtein(normalize_plate(x["patente_validada"]), normalize_plate(y["patente_validada"]))
        dframe = abs(int(x["frame_idx"]) - int(y["frame_idx"]))
        same_id = x.get("id_vehiculo") is not None and x.get("id_vehiculo") == y.get("id_vehiculo")
        if same_id or (edit <= max_edit and dframe <= near_frames):
            return (0 if same_id else 1, edit, dframe)
        return None

    different, only_a, only_b = _pair_greedy(rest_a, rest_b, near_cost)
    return {"matched": matched, "different_reading": different, "only_a": only_a, "only_b": only_b}


def summarize(result, frame_tolerance=5):
    """Cuenta y veredicto de una comparación."""
    diffs = [abs(int(x["frame_idx"]) - int(y["frame_idx"])) for x, y in result["matched"]]
    field_diffs = {k: 0 for k in COMPARED_FIELDS}
    for x, y in result["matched"]:
        for k in COMPARED_FIELDS:
            if x.get(k) != y.get(k):
                field_diffs[k] += 1
    over = [d for d in diffs if d > frame_tolerance]
    parity = (not result["only_a"] and not result["only_b"] and not result["different_reading"]
              and not over)
    return {
        "n_a": len(result["matched"]) + len(result["different_reading"]) + len(result["only_a"]),
        "n_b": len(result["matched"]) + len(result["different_reading"]) + len(result["only_b"]),
        "matched": len(result["matched"]),
        "different_reading": len(result["different_reading"]),
        "only_a": len(result["only_a"]), "only_b": len(result["only_b"]),
        "max_frame_diff": max(diffs) if diffs else 0,
        "frame_diff_over_tolerance": len(over),
        "fields_that_differ_in_matches": field_diffs,
        "parity": parity,
    }


def _fmt(d, k):
    v = d.get(k)
    return "-" if v is None else str(v)


def format_report(name, result, summary, label_a, label_b, frame_tolerance):
    lines = [f"== {name} ==",
             f"  {label_a}: {summary['n_a']} detecciones | {label_b}: {summary['n_b']} detecciones",
             f"  coinciden={summary['matched']}  lectura distinta={summary['different_reading']}  "
             f"solo {label_a}={summary['only_a']}  solo {label_b}={summary['only_b']}",
             f"  desfase máximo de frame entre coincidencias: {summary['max_frame_diff']} "
             f"(tolerancia {frame_tolerance}; {summary['frame_diff_over_tolerance']} la superan)"]
    changed = {k: v for k, v in summary["fields_that_differ_in_matches"].items() if v}
    if changed:
        lines.append("  campos que difieren en las coincidencias: " +
                     ", ".join(f"{k}={v}" for k, v in changed.items()))
    if result["different_reading"]:
        lines.append(f"  -- misma detección, distinta patente ({label_a} -> {label_b}):")
        for x, y in result["different_reading"]:
            lines.append(f"     ID {_fmt(x, 'id_vehiculo')}/{_fmt(y, 'id_vehiculo')}  "
                         f"{x['patente_validada']} (F{x['frame_idx']}) -> {y['patente_validada']} (F{y['frame_idx']})")
    for tag, label, items in (("only_a", label_a, result["only_a"]), ("only_b", label_b, result["only_b"])):
        if items:
            lines.append(f"  -- solo en {label}:")
            for x in items:
                lines.append(f"     {x['patente_validada']}  F{x['frame_idx']}  ID {_fmt(x, 'id_vehiculo')}  "
                             f"conf={_fmt(x, 'confianza_ocr_promedio')}  {_fmt(x, 'finalization')}")
    lines.append("  => " + ("PARIDAD" if summary["parity"] else "CON DIFERENCIAS"))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--a", nargs="+", required=True, help="Detecciones de la corrida A (p. ej. V5)")
    p.add_argument("--b", nargs="+", required=True, help="Detecciones de la corrida B (p. ej. modular)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--frame-tolerance", type=int, default=5)
    p.add_argument("--max-edit", type=int, default=2)
    p.add_argument("--near-frames", type=int, default=150)
    p.add_argument("--out", default=None, help="Guarda el resultado completo en JSON")
    args = p.parse_args(argv)

    if len(args.a) != len(args.b):
        p.error("--a y --b necesitan la misma cantidad de archivos (se emparejan por orden)")

    report, all_ok = {}, True
    for path_a, path_b in zip(args.a, args.b):
        name = os.path.basename(path_b)
        result = compare_detections(load_detections(path_a), load_detections(path_b),
                                    args.max_edit, args.near_frames)
        summary = summarize(result, args.frame_tolerance)
        all_ok &= summary["parity"]
        print(format_report(name, result, summary, args.label_a, args.label_b, args.frame_tolerance))
        report[name] = {"a": path_a, "b": path_b, "summary": summary, **result}

    if len(args.a) > 1:
        print(f"\nTOTAL: {'PARIDAD en todos los videos' if all_ok else 'hay videos CON DIFERENCIAS'}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"label_a": args.label_a, "label_b": args.label_b,
                       "frame_tolerance": args.frame_tolerance, "parity": all_ok, "videos": report},
                      f, ensure_ascii=False, indent=2)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
