"""Pipeline ALPR en tiempo real: RF-DETR -> ByteTrack -> RT-DETR (batcheado) -> OCR -> consenso.

Cambios de estructura respecto de pipelineV5.py (la lógica de decisión no cambia):
  - Recibe `ModelBundle`, `PipelineOptions` y `Thresholds` por constructor: nada
    de `args` ni de constantes globales pisadas desde la CLI.
  - `run_video()` es reutilizable: resetea el estado por video y devuelve un
    `RunResult` con las detecciones; escribir JSON/video es opcional.
  - Modo `record_observations`: no finaliza nada y graba cada voto para poder
    re-decidir los umbrales de consenso offline (alpr.replay) sin volver a
    correr los modelos.
  - `snapshots_dir=None` desactiva la escritura de evidencia en disco
    (calibración/evaluación masiva).
"""
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
import torch

from .contracts import sanitize_video_stem, vehicle_crop_filename, write_image_atomic
from .config import (
    CONSENSUS_REASON, MIN_VEHICLE_CROP_HEIGHT, MIN_VEHICLE_CROP_WIDTH,
    OCR_INTERVAL_FRAMES, OCR_VARIANT_BASE, OCR_VARIANT_ORDER, OCR_ZONE_START_RATIO,
    PipelineOptions, Thresholds, TRACK_TTL_FRAMES,
)
from .finalize import (
    OUTCOME_DUPLICATE, OUTCOME_OK, audit_records, prepare_record, snapshot_name,
)
from .imaging import has_plate_characteristics
from .ocr import PlateOCR, new_ocr_stats
from .tracking import OcrJob, TrackState, VehicleTrack


@dataclass
class RunResult:
    source: str
    detections: list
    stats: dict
    frames: int
    fps: float
    elapsed: float
    audit_problems: list = field(default_factory=list)
    observations: dict = None   # solo en modo record_observations
    discards: list = field(default_factory=list)   # qué se descartó y por qué


class ALPRPipeline:
    def __init__(self, models, options=None, thresholds=None, record_observations=False):
        self.models = models
        self.options = options or PipelineOptions()
        self.th = thresholds or Thresholds()
        self.record = bool(record_observations)
        self.device = models.device

        self.strict_plate = bool(self.options.strict_plate)
        self.flush_unfinalized = bool(self.options.flush_unfinalized)
        self.rtdetr_batch_size = max(1, int(self.options.rtdetr_batch))
        self._rtdetr_batch_effective = self.rtdetr_batch_size

        # En modo grabación se registran TODAS las lecturas de formato válido
        # (piso 0.0): el piso real (min_avg_ocr_conf) se decide offline.
        # `self.stats` es UN dict compartido con el OCR; se reinicia in situ por video.
        self.stats = {}
        self.ocr = PlateOCR(
            models.ocr_reader, self.th, self.strict_plate,
            dual_ocr=self.options.dual_ocr, max_attempts=self.options.ocr_max_attempts,
            candidate_min_conf=0.0 if self.record else None, stats=self.stats)

        self.debug_dir = None
        self.export_dir = self.options.export_vehicle_dir
        if self.export_dir:
            os.makedirs(self.export_dir, exist_ok=True)
        self.video_stem = "video"
        self.discard_dir = None
        if self.options.snapshots_dir:
            os.makedirs(self.options.snapshots_dir, exist_ok=True)
            if self.options.debug_crops:
                self.debug_dir = os.path.join(self.options.snapshots_dir, "debug")
                os.makedirs(self.debug_dir, exist_ok=True)
            if self.options.log_discards:
                self.discard_dir = os.path.join(self.options.snapshots_dir, "descartes")
                os.makedirs(self.discard_dir, exist_ok=True)

        self._reset_state()

    # --------------------------------------------------------------------------
    # Estado por video
    # --------------------------------------------------------------------------
    def _reset_state(self):
        self.tracks = {}
        self.json_log = []
        self.recent_plates = {}
        self._jsonl_handle = None
        self._observations = {}
        self.discards = []
        self._track_outcome = {}   # track_id -> registrada | duplicada | rechazada
        self._rtdetr_batch_effective = self.rtdetr_batch_size
        self.stats.clear()
        self.stats.update({
            "vehicle_crops_skipped": 0,   # descartados antes de RT-DETR
            "plate_crops_total": 0,       # crops de chapa producidos por RT-DETR
            "plate_crops_skipped": 0,     # descartados por el focus mask
            "plate_crops_to_ocr": 0,      # llegaron al OCR
            "focus_mask_rejections": defaultdict(int),
            "rtdetr_forwards": 0,
            "rtdetr_crops_in_batches": 0,
            "rtdetr_oom_backoffs": 0,
            "tracks_flushed": 0,
            "records_consensus": 0,
            "records_recovered": 0,
            "records_format_invalid": 0,
            "records_rejected_invalid": 0,
            "records_duplicate": 0,
            "vehicle_crops_exported": 0,   # recortes de vehículo escritos para el clasificador
            "export_audit_violations": 0,
        })
        self.stats.update(new_ocr_stats())

    def _init_video(self, source):
        self.video_info = sv.VideoInfo.from_video_path(source)
        self.roi_y = int(self.video_info.resolution_wh[1] * OCR_ZONE_START_RATIO)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.3, lost_track_buffer=30,
            frame_rate=self.video_info.fps)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

    # --------------------------------------------------------------------------
    # Detección
    # --------------------------------------------------------------------------
    def _detect_vehicles(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw = self.models.vehicle_model.predict(frame_rgb, threshold=self.th.confidence_vehicle)
        if hasattr(raw, 'xyxy') and len(raw.xyxy) > 0:
            dets = sv.Detections(
                xyxy=np.array(raw.xyxy),
                confidence=np.array(raw.confidence),
                class_id=np.array(raw.class_id).astype(int),
            )
            return dets[np.isin(dets.class_id, list(self.models.vehicle_class_names.keys()))]
        return sv.Detections.empty()

    def _plate_forward(self, vehicle_crops):
        """Forward de RT-DETR sobre N recortes de vehículo -> lista de box|None.

        [FIX-v3 #4] Forward y post-proceso dentro del mismo inference_mode, y lo
        que sale ya es numpy (los inference tensors no se pueden usar afuera).
        target_sizes = (h, w) ORIGINALES de cada crop."""
        crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in vehicle_crops]
        proc = self.models.plate_processor
        inputs = proc(images=crops_rgb, return_tensors="pt").to(self.device)
        target_sizes = torch.tensor([c.shape[:2] for c in vehicle_crops]).to(self.device)

        batch_boxes = []
        with torch.inference_mode():
            if self.options.fp16 and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.models.plate_model(**inputs)
            else:
                outputs = self.models.plate_model(**inputs)

            results = proc.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.th.confidence_plate)
            for res in results:
                if len(res["scores"]) == 0:
                    batch_boxes.append(None)
                else:
                    scores = res["scores"].float().cpu().numpy()
                    boxes = res["boxes"].float().cpu().numpy()
                    batch_boxes.append(boxes[int(np.argmax(scores))])
        return batch_boxes

    def _detect_plate_box(self, vehicle_crop):
        """Camino secuencial (línea base V3): un forward por vehículo."""
        return self._plate_forward([vehicle_crop])[0]

    def _detect_plate_box_batch_safe(self, crops):
        """[BATCH-#2] Chunking con backoff ante CUDA OOM.

        [FIX-v5 #3] Solo se captura OOM (cualquier otro RuntimeError se
        re-lanza); el lote reducido se recuerda entre frames; es el único lugar
        donde se parte en chunks."""
        n = len(crops)
        results = [None] * n
        i = 0
        size = min(self._rtdetr_batch_effective, n)
        while i < n:
            chunk = crops[i:i + size]
            boxes = None
            oom = False
            try:
                boxes = self._plate_forward(chunk)
            except RuntimeError as e:
                is_oom = (isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ()))
                          or "out of memory" in str(e).lower())
                if not is_oom:
                    raise
                oom = True

            if oom:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if size <= 1:
                    logging.error("[batch] OOM incluso con batch=1: se descartan los "
                                  "crops restantes de este frame.")
                    break
                size = max(1, size // 2)
                self._rtdetr_batch_effective = min(self._rtdetr_batch_effective, size)
                self.stats["rtdetr_oom_backoffs"] += 1
                logging.warning(f"[batch] OOM de VRAM en RT-DETR: reintento con batch={size}")
                continue

            self.stats["rtdetr_forwards"] += 1
            self.stats["rtdetr_crops_in_batches"] += len(chunk)
            for k, b in enumerate(boxes):
                results[i + k] = b
            i += len(chunk)
        return results

    # --------------------------------------------------------------------------
    # Votos y finalización
    # --------------------------------------------------------------------------
    def _cast_vote(self, track, text, conf, frame_idx, plate_crop, processed, bbox, vehicle_crop=None):
        if self.record:
            # Se graba la lectura y NO se guardan imágenes (no hace falta y pesan).
            obs = self._observations.setdefault(int(track.track_id), {
                "vehicle_class": track.vehicle_class_name, "last_seen_frame": -1, "votes": []})
            obs["votes"].append({"frame": int(frame_idx), "text": text, "conf": float(conf),
                                 "bbox": [int(x) for x in bbox]})
            track.add_vote(text, conf, frame_idx, None, None, bbox)
        else:
            track.add_vote(text, conf, frame_idx, plate_crop, processed, bbox, vehicle_crop)

    def _try_finalize(self, v_track):
        """Finaliza si el consenso cumple los criterios. En modo grabación nunca
        finaliza: así se observan TODAS las lecturas del track."""
        if self.record or v_track.state != TrackState.OCR_ACTIVE:
            return False
        if v_track.check_finalization(self.th):
            self._finalize_track(v_track)
            return True
        return False

    def _snapshot_writer(self, plate, track_id, frame, crop, processed):
        if not self.options.snapshots_dir:
            return None
        name = snapshot_name(plate, track_id, frame)
        path = os.path.join(self.options.snapshots_dir, name)
        if not cv2.imwrite(path, crop):
            logging.warning(f"[export] No se pudo escribir el snapshot: {path}")
            return None
        # [MEJORA-v3 #11] natural vs. lo que vio el OCR (figuras de tesis / diagnóstico).
        if self.debug_dir and processed is not None and processed.size > 0:
            cv2.imwrite(os.path.join(
                self.debug_dir, f"{plate}_ID{track_id}_F{frame}_ocr.png"), processed)
        return path

    def _finalize_track(self, v_track, reason=CONSENSUS_REASON):
        """Devuelve True solo si se agregó un registro a json_log."""
        v_track.state = TrackState.FINALIZED
        best_plate, avg_conf, count = v_track.get_consensus_plate()
        v_track.best_plate_text = best_plate

        record, outcome, detail = prepare_record(
            v_track, reason, self.video_info.fps, self.strict_plate,
            self.recent_plates, self.options.dedup_seconds,
            snapshot_writer=self._snapshot_writer)

        if outcome == OUTCOME_DUPLICATE:
            self._track_outcome[int(v_track.track_id)] = "duplicada"
            self._note_discard(v_track.last_seen_frame, v_track.track_id, "patente_duplicada",
                               patente=best_plate, detalle=detail)
            self.stats["records_duplicate"] += 1
            logging.info(f"[DUPLICADO] ID #{v_track.track_id} -> {best_plate} ({detail}, no se registra)")
            return False
        if outcome != OUTCOME_OK:
            self._track_outcome[int(v_track.track_id)] = "rechazada"
            self._note_discard(v_track.last_seen_frame, v_track.track_id, "registro_rechazado",
                               patente=best_plate, detalle=detail)
            self.stats["records_rejected_invalid"] += 1
            logging.error(f"[export] ID #{v_track.track_id} NO se registra "
                          f"(patente={best_plate!r}): {detail}")
            return False

        self.json_log.append(record)
        self._track_outcome[int(v_track.track_id)] = "registrada"
        self._append_jsonl(record)
        self._export_vehicle_crop(v_track, best_plate)
        if reason == CONSENSUS_REASON:
            self.stats["records_consensus"] += 1
            tag = "NUEVO ALPR"
        else:
            self.stats["records_recovered"] += 1
            tag = "RECUPERADO SIN CONSENSO"
        if not record["format_valid"]:
            self.stats["records_format_invalid"] += 1
        logging.info(f"[{tag}] ID #{v_track.track_id} ({v_track.vehicle_class_name}) -> "
                     f"{best_plate} (Conf: {avg_conf:.2f}, Votos: {count}, {reason}, "
                     f"format_valid={record['format_valid']})")
        return True

    def _export_vehicle_crop(self, v_track, plate):
        """Productor del contrato con alpr.classify: escribe el recorte del vehículo (el de la
        MISMA lectura que respalda la patente) como vehiculo_<track_id>__<video>.jpg, de forma
        atómica. Un fallo acá se registra pero nunca afecta la detección."""
        if not self.export_dir:
            return
        ev = v_track.evidence.get(plate)
        if ev is None or ev.vehicle_crop is None or ev.vehicle_crop.size == 0:
            logging.warning(f"[export-vehiculo] ID #{v_track.track_id}: sin recorte de vehículo para exportar")
            return
        path = os.path.join(self.export_dir, vehicle_crop_filename(v_track.track_id, self.video_stem))
        try:
            write_image_atomic(path, ev.vehicle_crop)
            self.stats["vehicle_crops_exported"] += 1
        except Exception as e:
            logging.warning(f"[export-vehiculo] ID #{v_track.track_id}: no se pudo escribir {path}: {e}")

    def _append_jsonl(self, record):
        """Persistencia incremental: si el proceso muere, no se pierde todo."""
        if self._jsonl_handle is None:
            return
        try:
            self._jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_handle.flush()
        except Exception as e:
            logging.warning(f"[jsonl] No se pudo volcar la detección incremental: {e}")

    def _flush_track(self, v_track, reason):
        """[FIX-v5 #4] Finaliza un track que se cierra SIN consenso (criterio en
        VehicleTrack.flush_eligible). Queda marcado en `finalization`."""
        if self.record or not self.flush_unfinalized or v_track.state != TrackState.OCR_ACTIVE:
            return False
        if not v_track.flush_eligible(self.th):
            return False
        recorded = self._finalize_track(v_track, reason=reason)
        if recorded:
            self.stats["tracks_flushed"] += 1
        return recorded

    def _flush_all_tracks(self, reason):
        for v_track in list(self.tracks.values()):
            self._flush_track(v_track, reason)

    def _purge_stale_tracks(self, current_frame, active_ids):
        stale_ids = [
            tid for tid, t in self.tracks.items()
            if (current_frame - t.last_seen_frame) > TRACK_TTL_FRAMES and tid not in active_ids
        ]
        for tid in stale_ids:
            self._flush_track(self.tracks[tid], "track_lost")
            self._log_track_discard(self.tracks[tid])
            self._note_track_end(self.tracks[tid])
            del self.tracks[tid]

    # --------------------------------------------------------------------------
    # Registro de descartes
    # --------------------------------------------------------------------------
    def _note_discard(self, frame_idx, track_id, reason, crop=None, **extra):
        """Anota un descarte (y guarda el recorte si hay carpeta de snapshots)."""
        if not self.options.log_discards:
            return
        rec = {"frame_idx": None if frame_idx is None else int(frame_idx),
               "id_vehiculo": None if track_id is None else int(track_id), "motivo": reason}
        rec.update(extra)
        if crop is not None and self.discard_dir is not None and crop.size > 0:
            name = f"F{int(frame_idx)}_ID{int(track_id)}_{reason.split(':')[0]}_{len(self.discards)}.jpg"
            path = os.path.join(self.discard_dir, name)
            rec["recorte"] = path if cv2.imwrite(path, crop) else None
        self.discards.append(rec)

    def _log_track_discard(self, v_track):
        """Explica por qué un vehículo terminó SIN ninguna detección registrada."""
        tid = int(v_track.track_id)
        if self.record or not self.options.log_discards or tid in self._track_outcome:
            return
        self._track_outcome[tid] = "descartado"
        if v_track.state == TrackState.NEW:
            reason, extra = "vehiculo_nunca_cruzo_zona_ocr", {}
        elif not v_track.votes:
            reason, extra = "vehiculo_sin_lecturas_validas", {}
        else:
            reason = "vehiculo_sin_consenso_no_recuperable"
            extra = {"lecturas": {p: {"votos": v_track.vote_counts[p],
                                      "confianza_media": round(v_track.votes[p] / v_track.vote_counts[p], 3)}
                                  for p in v_track.votes}}
        self._note_discard(v_track.last_seen_frame, tid, reason,
                           tipo_vehiculo=v_track.vehicle_class_name, **extra)

    def _note_track_end(self, v_track):
        if self.record and int(v_track.track_id) in self._observations:
            self._observations[int(v_track.track_id)]["last_seen_frame"] = int(v_track.last_seen_frame)

    # --------------------------------------------------------------------------
    # FASE A: máquina de estados y recolección de jobs (CPU)
    # --------------------------------------------------------------------------
    def _get_or_create_track(self, track_id, vehicle_class_name):
        if track_id not in self.tracks:
            self.tracks[track_id] = VehicleTrack(track_id, vehicle_class_name)
        return self.tracks[track_id]

    def _maybe_activate_ocr(self, v_track, bbox_int):
        """NEW -> OCR_ACTIVE cuando el CENTRO del bbox cruza roi_y (una vez).
        (Reconstrucción del criterio de V3; ver nota en el README.)"""
        x1, y1, x2, y2 = bbox_int
        if 0.5 * (y1 + y2) >= self.roi_y:
            v_track.state = TrackState.OCR_ACTIVE

    def _collect_ocr_jobs(self, frame, tracked, frame_idx):
        jobs = []
        if tracked is None or len(tracked) == 0 or tracked.tracker_id is None:
            return jobs

        H, W = frame.shape[:2]
        names = self.models.vehicle_class_names
        for bbox, class_id, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
            if track_id is None:
                continue
            track_id = int(track_id)
            v_track = self._get_or_create_track(track_id, names.get(int(class_id), "vehicle"))
            v_track.last_seen_frame = frame_idx

            x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
            x2 = min(W, int(bbox[2])); y2 = min(H, int(bbox[3]))
            if x2 <= x1 or y2 <= y1:
                continue

            if v_track.state == TrackState.NEW:
                self._maybe_activate_ocr(v_track, (x1, y1, x2, y2))
            if v_track.state != TrackState.OCR_ACTIVE:
                continue

            if self._try_finalize(v_track):
                continue
            if (frame_idx - v_track.last_ocr_frame) < OCR_INTERVAL_FRAMES:
                continue

            vehicle_crop = frame[y1:y2, x1:x2]
            vh, vw = vehicle_crop.shape[:2]
            if not self.options.disable_focus_mask and (
                    vw < MIN_VEHICLE_CROP_WIDTH or vh < MIN_VEHICLE_CROP_HEIGHT):
                self.stats["vehicle_crops_skipped"] += 1
                v_track.last_ocr_frame = frame_idx
                continue

            jobs.append(OcrJob(track=v_track, crop=vehicle_crop,
                               vehicle_class=v_track.vehicle_class_name,
                               bbox=np.array([x1, y1, x2, y2], dtype=np.int32)))
            # El intervalo corre desde el INTENTO (encolado), no desde el voto.
            v_track.last_ocr_frame = frame_idx
        return jobs

    # --------------------------------------------------------------------------
    # FASE B: forwards batcheados + OCR por job
    # --------------------------------------------------------------------------
    def _consume_job(self, job, box_p, frame_idx):
        """Después del forward: clamp del box, recorte de chapa, focus mask de
        chapa, OCR adaptativo y UN voto como máximo ([OCR-ADAPT #4])."""
        tid = job.track.track_id
        if box_p is None:
            self._note_discard(frame_idx, tid, "rtdetr_sin_patente", tipo_vehiculo=job.vehicle_class)
            return
        vehicle_crop = job.crop
        ch, cw = vehicle_crop.shape[:2]
        px1 = max(0, min(int(box_p[0]), cw))
        py1 = max(0, min(int(box_p[1]), ch))
        px2 = max(0, min(int(box_p[2]), cw))
        py2 = max(0, min(int(box_p[3]), ch))
        if px2 <= px1 or py2 <= py1:
            self._note_discard(frame_idx, tid, "caja_de_patente_invalida", tipo_vehiculo=job.vehicle_class)
            return

        plate_crop = vehicle_crop[py1:py2, px1:px2]
        self.stats["plate_crops_total"] += 1

        if not self.options.disable_focus_mask:
            ok, reason = has_plate_characteristics(plate_crop, job.vehicle_class)
            if not ok:
                self.stats["plate_crops_skipped"] += 1
                self.stats["focus_mask_rejections"][reason] += 1
                self._note_discard(frame_idx, tid, "focus_mask", crop=plate_crop,
                                   detalle=reason, tipo_vehiculo=job.vehicle_class)
                return

        self.stats["plate_crops_to_ocr"] += 1
        result = self.ocr.read(plate_crop, job.vehicle_class)
        if result is None:
            self._note_discard(frame_idx, tid, "ocr_sin_lectura_valida", crop=plate_crop,
                               tipo_vehiculo=job.vehicle_class,
                               intentos=list(self.ocr.last_attempts))
            return
        text, conf, processed = result
        self._cast_vote(job.track, text, float(conf), frame_idx, plate_crop, processed, job.bbox,
                        vehicle_crop=job.crop if self.export_dir else None)
        self._try_finalize(job.track)

    def _run_ocr_jobs(self, jobs, frame_idx):
        if not jobs:
            return
        if self.options.sequential_rtdetr:
            for job in jobs:
                self.stats["rtdetr_forwards"] += 1
                self.stats["rtdetr_crops_in_batches"] += 1
                self._consume_job(job, self._detect_plate_box(job.crop), frame_idx)
            return
        boxes = self._detect_plate_box_batch_safe([job.crop for job in jobs])
        for job, box_p in zip(jobs, boxes):
            self._consume_job(job, box_p, frame_idx)

    # --------------------------------------------------------------------------
    # Anotación y loop principal
    # --------------------------------------------------------------------------
    def _draw_annotations(self, frame, tracked):
        labels = []
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for t_id in tracker_ids:
            if t_id is not None and t_id in self.tracks:
                v_track = self.tracks[t_id]
                if v_track.state == TrackState.FINALIZED:
                    display_text = v_track.best_plate_text
                else:
                    plate, _, _ = v_track.get_consensus_plate()
                    display_text = plate if plate else v_track.vehicle_class_name
                labels.append(f"#{t_id} | {display_text}")
            else:
                labels.append("...")
        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=tracked)
        annotated = self.label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
        cv2.line(annotated, (0, self.roi_y), (self.video_info.resolution_wh[0], self.roi_y),
                 (0, 0, 255), 2)
        cv2.putText(annotated, "ZONA LECTURA OCR", (10, self.roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated

    def _process_frames(self, source, output_video):
        frame_idx = 0
        t_start = time.time()
        gen = sv.get_video_frames_generator(source_path=source)
        sink = sv.VideoSink(target_path=output_video, video_info=self.video_info) if output_video else None
        if sink is not None:
            sink.__enter__()
        try:
            for frame in gen:
                if self.options.max_frames and frame_idx >= self.options.max_frames:
                    break
                detections = self._detect_vehicles(frame)
                tracked = self.tracker.update_with_detections(detections)

                jobs = self._collect_ocr_jobs(frame, tracked, frame_idx)   # FASE A
                self._run_ocr_jobs(jobs, frame_idx)                        # FASE B

                tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
                active_ids = {int(t) for t in tracker_ids if t is not None}
                self._purge_stale_tracks(frame_idx, active_ids)

                if sink is not None:
                    sink.write_frame(self._draw_annotations(frame, tracked))

                frame_idx += 1
                if frame_idx % 100 == 0:
                    elapsed = max(time.time() - t_start, 1e-6)
                    logging.info(f"[progreso] {frame_idx} frames | {frame_idx / elapsed:.2f} FPS | "
                                 f"tracks activos={len(self.tracks)} | jobs último frame={len(jobs)}")
        finally:
            if sink is not None:
                sink.__exit__(None, None, None)
        return frame_idx, time.time() - t_start

    def run_video(self, source, output_json=None, output_video=None):
        """Procesa UN video. Reutilizable: resetea el estado en cada llamada.
        Devuelve un RunResult; `output_json` / `output_video` son opcionales."""
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        self._reset_state()
        self._init_video(source)
        self.video_stem = sanitize_video_stem(os.path.splitext(os.path.basename(source))[0])

        if output_json:
            d = os.path.dirname(output_json)
            if d:
                os.makedirs(d, exist_ok=True)
            self._jsonl_handle = open(os.path.splitext(output_json)[0] + ".jsonl", "w",
                                      encoding="utf-8")
        if output_video:
            d = os.path.dirname(output_video)
            if d:
                os.makedirs(d, exist_ok=True)

        try:
            frames, elapsed = self._process_frames(source, output_video)
            self._flush_all_tracks("end_of_video")
            for v_track in self.tracks.values():
                self._log_track_discard(v_track)
                self._note_track_end(v_track)
        finally:
            if self._jsonl_handle is not None:
                self._jsonl_handle.close()
                self._jsonl_handle = None

        problems = audit_records(self.json_log, self.strict_plate)
        self.stats["export_audit_violations"] = len(problems)

        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(self.json_log, f, ensure_ascii=False, indent=2)
            logging.info(f"[salida] {len(self.json_log)} detecciones -> {output_json}")

        stats = dict(self.stats)   # snapshot: el próximo run_video reinicia self.stats
        self._log_stats(frames, elapsed)
        self._log_discards()
        if output_json and self.options.log_discards:
            path = os.path.splitext(output_json)[0] + "_descartes.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.discards, f, ensure_ascii=False, indent=2)
            logging.info(f"[salida] {len(self.discards)} descartes -> {path}")

        observations = None
        if self.record:
            observations = {
                "source": os.path.basename(source),
                "fps": float(self.video_info.fps),
                "total_frames": int(frames),
                "strict_plate": self.strict_plate,
                "ocr": {"dual_ocr": self.options.dual_ocr,
                        "max_attempts": self.options.ocr_max_attempts,
                        "disable_focus_mask": self.options.disable_focus_mask},
                "detection_thresholds": {"confidence_vehicle": self.th.confidence_vehicle,
                                         "confidence_plate": self.th.confidence_plate},
                "tracks": {str(k): v for k, v in self._observations.items()},
            }
        return RunResult(source=source, detections=list(self.json_log), stats=stats,
                         frames=frames, fps=float(self.video_info.fps), elapsed=elapsed,
                         audit_problems=problems, observations=observations,
                         discards=list(self.discards))

    def _log_discards(self):
        """Resumen legible de los descartes: por motivo y, por vehículo, por qué no
        salió ninguna patente."""
        if not self.options.log_discards or not self.discards:
            return
        by_reason = defaultdict(int)
        for d in self.discards:
            by_reason[d["motivo"]] += 1
        logging.info("[DESCARTES] Por motivo: %s", ", ".join(
            f"{k}={v}" for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1])))
        for d in self.discards:
            if d["motivo"].startswith("vehiculo_"):
                extra = f" lecturas={d['lecturas']}" if "lecturas" in d else ""
                logging.info("[DESCARTES] ID #%s (%s) sin patente registrada: %s%s",
                             d["id_vehiculo"], d.get("tipo_vehiculo"), d["motivo"], extra)
        reads = [(d, a) for d in self.discards if d["motivo"] == "ocr_sin_lectura_valida"
                 for a in d["intentos"] if a["texto_leido"]]
        if reads:
            logging.info("[DESCARTES] Textos leídos y descartados (formato/confianza): %s", ", ".join(
                sorted({f"{a['texto_leido']}->{a['texto_corregido']}" for _, a in reads})))

    # --------------------------------------------------------------------------
    # Métricas
    # --------------------------------------------------------------------------
    def _log_stats(self, frames, elapsed):
        s = self.stats
        total = s["plate_crops_total"]
        ahorro = (100.0 * s["plate_crops_skipped"] / total) if total else 0.0
        logging.info("[MÉTRICAS] %d frames en %.1fs (%.2f FPS)", frames, elapsed,
                     frames / max(elapsed, 1e-6))
        logging.info("[MÉTRICAS] Crops de chapa: %d | descartados por focus mask: %d (%.1f%%) | "
                     "enviados a OCR: %d", total, s["plate_crops_skipped"], ahorro,
                     s["plate_crops_to_ocr"])
        logging.info("[MÉTRICAS] Vehículos descartados ANTES de RT-DETR: %d", s["vehicle_crops_skipped"])

        crops = s["ocr_crops_processed"]
        if crops:
            order = (OCR_VARIANT_BASE,) + OCR_VARIANT_ORDER
            por_variante = ", ".join(f"{v}={s['ocr_attempts_by_variant'].get(v, 0)}" for v in order)
            ganadoras = ", ".join(f"{v}={s['ocr_selected_by_variant'].get(v, 0)}" for v in order)
            logging.info("[MÉTRICAS][OCR] Crops: %d | con candidato final: %d (%.1f%%) | sin lectura válida: %d",
                         crops, s["ocr_crops_resolved"], 100.0 * s["ocr_crops_resolved"] / crops,
                         crops - s["ocr_crops_resolved"])
            logging.info("[MÉTRICAS][OCR] Intentos: %d (%.2f por crop, máx %d) | por variante: %s",
                         s["ocr_attempts_total"], s["ocr_attempts_total"] / crops,
                         1 + len(self.ocr.variants), por_variante)
            logging.info("[MÉTRICAS][OCR] Válidas: %d | rechazadas por formato: %d | por confianza: %d | "
                         "sin texto: %d", s["ocr_valid_readings"], s["ocr_rejected_format"],
                         s["ocr_rejected_confidence"], s["ocr_no_text"])
            logging.info("[MÉTRICAS][OCR] Variante ganadora por crop: %s", ganadoras)

        logging.info("[MÉTRICAS] Detecciones: %d | por vaciado: %d | duplicadas: %d | rechazadas por "
                     "evidencia inconsistente: %d | violaciones de auditoría: %d",
                     len(self.json_log), s["tracks_flushed"], s["records_duplicate"],
                     s["records_rejected_invalid"], s["export_audit_violations"])
        logging.info("[MÉTRICAS][POLÍTICA] Consenso: %d | RECUPERADAS sin consenso: %d | format_valid=False: %d | "
                     "validación: %s", s["records_consensus"], s["records_recovered"],
                     s["records_format_invalid"],
                     "ESTRICTA" if self.strict_plate else "PERMISIVA/DEBUG (no equivale a formato argentino)")
        if s["rtdetr_forwards"]:
            logging.info("[MÉTRICAS][BATCH] Forwards RT-DETR: %d | crops/forward medio: %.2f | "
                         "backoffs OOM: %d | lote efectivo final: %d", s["rtdetr_forwards"],
                         s["rtdetr_crops_in_batches"] / s["rtdetr_forwards"],
                         s["rtdetr_oom_backoffs"], self._rtdetr_batch_effective)
        if s["focus_mask_rejections"]:
            resumen = ", ".join(f"{k}={v}" for k, v in
                                sorted(s["focus_mask_rejections"].items(), key=lambda kv: -kv[1]))
            logging.info("[MÉTRICAS] Rechazos de focus mask por motivo: %s", resumen)
        if self.device == "cuda":
            logging.info("[MÉTRICAS][VRAM] Pico del proceso: %.2f GB",
                         torch.cuda.max_memory_allocated() / 1024**3)
