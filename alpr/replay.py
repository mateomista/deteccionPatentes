"""Re-decisión OFFLINE sobre observaciones grabadas (sin modelos, sin GPU).

El pipeline en modo `record_observations` guarda, por cada track, todas las
lecturas de formato válido (frame, texto, confianza). Acá se re-aplica la
política de consenso/recuperación con CUALQUIER combinación de umbrales de
decisión (`DECISION_FIELDS` de config.py) en milisegundos: barrer 1000+
combinaciones cuesta lo que cuesta correr Python, no 1000 pasadas de RF-DETR +
RT-DETR + EasyOCR.

Reutiliza `VehicleTrack` y `prepare_record`, o sea la MISMA lógica de decisión
que el pipeline real. Diferencias conocidas (por eso la evaluación final se
hace con una corrida real, no con el replay):
  - En la corrida real, el OCR descarta un candidato bajo `min_avg_ocr_conf`
    ANTES de elegir el ganador del crop (preferencia por formato oficial). Acá
    se filtra después. Solo difiere en modo permisivo y en bordes raros.
  - La corrida real deja de leer un track cuando finaliza, lo que cambia a qué
    frames cae el intervalo de OCR; el replay ve el superconjunto de lecturas.
"""
from .config import CONSENSUS_REASON, TRACK_TTL_FRAMES
from .finalize import OUTCOME_OK, prepare_record
from .tracking import TrackState, VehicleTrack


def replay_video(obs, th, flush_unfinalized=True, dedup_seconds=30.0):
    """Devuelve la lista de registros de detección que el pipeline habría
    producido sobre `obs` con los umbrales `th`."""
    fps = obs["fps"]
    strict = obs["strict_plate"]
    total_frames = obs["total_frames"]

    events = []   # (frame de disparo, track_id, track, razón)
    for tid_str, t in obs["tracks"].items():
        tid = int(tid_str)
        track = VehicleTrack(tid, t["vehicle_class"])
        track.state = TrackState.OCR_ACTIVE
        finalized = False
        for v in t["votes"]:
            if v["conf"] < th.min_avg_ocr_conf:      # piso por candidato
                continue
            track.add_vote(v["text"], v["conf"], v["frame"], None, None, v["bbox"])
            if track.check_finalization(th):
                events.append((v["frame"], tid, track, CONSENSUS_REASON))
                finalized = True
                break
        if finalized or not flush_unfinalized or not track.votes:
            continue
        if not track.flush_eligible(th):
            continue
        # El pipeline purga un track cuando (frame - last_seen) > TTL.
        purge_frame = t["last_seen_frame"] + TRACK_TTL_FRAMES + 1
        if purge_frame < total_frames:
            events.append((purge_frame, tid, track, "track_lost"))
        else:
            events.append((total_frames, tid, track, "end_of_video"))

    events.sort(key=lambda e: (e[0], e[1]))
    recent_plates = {}
    records = []
    for _, _, track, reason in events:
        track.state = TrackState.FINALIZED
        record, outcome, _ = prepare_record(
            track, reason, fps, strict, recent_plates, dedup_seconds,
            snapshot_writer=None, require_crop=False)
        if outcome == OUTCOME_OK:
            records.append(record)
    return records
