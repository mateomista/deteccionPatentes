"""Construcción y auditoría de registros de detección ([EXPORT]).

Lógica PURA y compartida: la usan tanto `ALPRPipeline` (corrida real) como
`alpr.replay` (re-decisión offline sobre observaciones grabadas). Que sea una
sola implementación garantiza que lo que se calibra offline se decide igual
que en producción.
"""
import logging
import os

from .config import EXPORT_FIELDS, FINALIZATION_REASONS
from .plate_text import is_valid_plate, matches_official_format

OUTCOME_OK = "ok"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_REJECTED = "rejected"


def evidence_problem(ev, plate, require_crop=True):
    """[EXPORT #2] Motivo (str) si la evidencia NO sirve para exportar `plate`,
    o None si es consistente. `require_crop=False` solo en el replay offline,
    donde no se guardan imágenes."""
    if ev is None:
        return "sin evidencia para la patente ganadora"
    if ev.text != plate:
        return f"evidencia de otra patente ({ev.text!r} != {plate!r})"
    if ev.frame is None or ev.frame < 0:
        return f"frame_idx inválido ({ev.frame})"
    if require_crop and (ev.crop is None or ev.crop.size == 0):
        return "sin recorte natural"
    if ev.bbox is None or len(ev.bbox) != 4:
        return "sin bounding box"
    return None


def prepare_record(track, reason, fps, strict, recent_plates, dedup_seconds,
                   snapshot_writer=None, require_crop=True):
    """Valida un track y arma su registro de exportación.

    Devuelve (record | None, outcome, detail). `outcome` es OUTCOME_OK,
    OUTCOME_DUPLICATE u OUTCOME_REJECTED; `detail` explica el rechazo.
    Muta `recent_plates` (deduplicación de tracks re-identificados).
    `snapshot_writer(plate, track_id, frame, crop, processed) -> path | None`.
    """
    best_plate, avg_conf, count = track.get_consensus_plate()

    # [EXPORT #3] Solo se exporta una patente que pase la validación configurada.
    if not best_plate or not is_valid_plate(best_plate, strict):
        return None, OUTCOME_REJECTED, "no pasa la validación de patente configurada"

    # .get(): evidence es un defaultdict y [] crearía una entrada vacía.
    ev = track.evidence.get(best_plate)
    problem = evidence_problem(ev, best_plate, require_crop=require_crop)
    if problem is None and not (fps and fps > 0):
        problem = f"fps inválido ({fps})"
    if problem is not None:
        return None, OUTCOME_REJECTED, problem

    timestamp = ev.frame / float(fps)   # >= 0 por construcción

    # Deduplicación: si ByteTrack pierde y recupera el mismo vehículo aparecen
    # dos IDs con la misma patente.
    if dedup_seconds > 0:
        last_seen = recent_plates.get(best_plate)
        if last_seen is not None and (timestamp - last_seen) <= dedup_seconds:
            recent_plates[best_plate] = timestamp
            return None, OUTCOME_DUPLICATE, f"visto hace {timestamp - last_seen:.1f}s"
    recent_plates[best_plate] = timestamp

    snap_path = None
    if snapshot_writer is not None and ev.crop is not None:
        snap_path = snapshot_writer(best_plate, track.track_id, ev.frame, ev.crop, ev.processed)

    record = {
        "id_vehiculo": int(track.track_id),
        "tipo_vehiculo": track.vehicle_class_name,
        "patente_validada": best_plate,
        "confianza_ocr_promedio": round(float(avg_conf), 3),
        "cantidad_votos": int(count),
        "timestamp_seconds": round(float(timestamp), 2),
        "frame_idx": int(ev.frame),
        "snapshot_path": snap_path,
        "bounding_box": [int(x) for x in ev.bbox],
        "finalization": reason,
        "format_valid": bool(matches_official_format(best_plate)),
    }
    return record, OUTCOME_OK, None


def snapshot_name(plate, track_id, frame):
    return f"{plate}_ID{track_id}_F{frame}.jpg"


def audit_records(records, strict):
    """[EXPORT #5] Verifica TODAS las detecciones. Devuelve la lista de
    violaciones (vacía en una corrida sana)."""
    problems = []
    for rec in records:
        tag = f"ID #{rec.get('id_vehiculo')} ({rec.get('patente_validada')})"
        missing = [k for k in EXPORT_FIELDS if k not in rec]
        if missing:
            problems.append(f"{tag}: faltan campos {missing}")
            continue
        if not isinstance(rec["timestamp_seconds"], (int, float)) or rec["timestamp_seconds"] < 0:
            problems.append(f"{tag}: timestamp_seconds inválido ({rec['timestamp_seconds']})")
        if not isinstance(rec["frame_idx"], int) or rec["frame_idx"] < 0:
            problems.append(f"{tag}: frame_idx inválido ({rec['frame_idx']})")
        bbox = rec["bounding_box"]
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)):
            problems.append(f"{tag}: bounding_box inválido ({bbox})")
        if not is_valid_plate(rec["patente_validada"], strict):
            problems.append(f"{tag}: patente no pasa la validación configurada")
        if rec["finalization"] not in FINALIZATION_REASONS:
            problems.append(f"{tag}: finalization desconocido ({rec['finalization']})")
        snap = rec["snapshot_path"]
        if snap is not None:
            name = os.path.basename(snap)
            if not name.startswith(f"{rec['patente_validada']}_ID{rec['id_vehiculo']}_F"):
                problems.append(f"{tag}: snapshot de otra patente/track ({name})")
            if not name.endswith(f"_F{rec['frame_idx']}.jpg"):
                problems.append(f"{tag}: snapshot de otro frame ({name})")
    for p in problems:
        logging.error(f"[export][AUDITORÍA] {p}")
    return problems
