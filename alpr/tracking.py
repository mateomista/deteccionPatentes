"""Estado por vehículo: votos, evidencia y criterio de consenso."""
from collections import defaultdict
from enum import Enum

from .plate_text import matches_official_format


class TrackState(Enum):
    NEW = "NEW"
    OCR_ACTIVE = "OCR_ACTIVE"
    FINALIZED = "FINALIZED"


class PlateEvidence:
    """Mejor evidencia visual asociada específicamente a UN texto de patente."""
    def __init__(self):
        self.best_conf = 0.0
        self.text = None            # [EXPORT #2] patente a la que pertenece esta evidencia
        self.crop = None            # recorte NATURAL (lo que vio la cámara)
        self.processed = None       # [MEJORA-v3 #11] lo que vio el OCR
        self.frame = -1
        self.bbox = None


class VehicleTrack:
    """Centraliza todo el estado y la evidencia asociada a un track_id."""
    def __init__(self, track_id, vehicle_class_name):
        self.track_id = track_id
        self.vehicle_class_name = vehicle_class_name
        self.state = TrackState.NEW
        self.last_seen_frame = -1
        self.last_ocr_frame = -999

        self.votes = {}        # {plate_text: sum_confidence}
        self.vote_counts = {}  # {plate_text: count}
        self.evidence = defaultdict(PlateEvidence)

        self.best_plate_text = None

    def add_vote(self, text, confidence, frame_idx, plate_crop, processed_crop, vehicle_bbox):
        self.votes[text] = self.votes.get(text, 0.0) + float(confidence)
        self.vote_counts[text] = self.vote_counts.get(text, 0) + 1

        ev = self.evidence[text]
        if confidence > ev.best_conf:
            ev.best_conf = float(confidence)
            ev.text = text      # [EXPORT #2] la evidencia declara a qué patente pertenece
            ev.crop = plate_crop.copy() if plate_crop is not None else None
            ev.processed = processed_crop.copy() if processed_crop is not None else None
            ev.frame = frame_idx
            ev.bbox = vehicle_bbox.copy() if vehicle_bbox is not None else None

    def get_consensus_plate(self):
        """Patente ganadora por sumatoria total de confianza."""
        if not self.votes:
            return None, 0.0, 0
        best_plate = max(self.votes, key=self.votes.get)
        count = self.vote_counts[best_plate]
        avg_conf = self.votes[best_plate] / count
        return best_plate, avg_conf, count

    def check_finalization(self, th):
        """`th`: Thresholds. Los umbrales llegan por parámetro (no globales)."""
        if self.state == TrackState.FINALIZED:
            return True
        best_plate, avg_conf, count = self.get_consensus_plate()
        if not best_plate:
            return False

        best_score = self.votes[best_plate]
        margin = 1.0
        if len(self.votes) > 1:
            sorted_plates = sorted(self.votes.keys(), key=self.votes.get, reverse=True)
            second_score = self.votes[sorted_plates[1]]
            margin = (best_score - second_score) / best_score if best_score > 0 else 0.0

        # [FIX-v3 #6] La vía rápida ahora saltea el margen; la estándar lo exige.
        if count >= th.early_finalize_votes and avg_conf >= th.early_finalize_conf:
            return True

        is_standard = count >= th.min_votes and avg_conf >= th.min_avg_ocr_conf
        return bool(is_standard and margin >= th.min_consensus_margin)

    def flush_eligible(self, th):
        """[FIX-v5 #4] ¿Se puede recuperar este track SIN consenso (track perdido
        o fin de video)? Criterio deliberadamente más conservador que el normal:
          - alguna lectura con confianza media >= min_avg_ocr_conf;
          - con >= min_votes votos se acepta el ganador aunque falte margen;
          - con UNA sola lectura, solo si cumple el formato oficial Y supera
            flush_single_vote_conf.
        Es una decisión pura: la comparten el pipeline y el replay offline.
        """
        plate, avg_conf, count = self.get_consensus_plate()
        if not plate or avg_conf < th.min_avg_ocr_conf:
            return False
        if count < th.min_votes:
            return bool(matches_official_format(plate)
                        and avg_conf >= th.flush_single_vote_conf)
        return True


class OcrJob:
    """[BATCH-#1] Candidato a detección de patente, producido en FASE A.

    Encapsula TODO lo que _consume_job necesita para replicar el camino de
    _process_plate_ocr de pipelineV3.py, de modo que la única diferencia entre
    el camino secuencial y el batcheado sea DÓNDE se ejecuta el forward de
    RT-DETR, nunca qué lógica se aplica después.
    """
    __slots__ = ("track", "crop", "vehicle_class", "bbox")

    def __init__(self, track, crop, vehicle_class, bbox):
        self.track = track                    # VehicleTrack asociado
        self.crop = crop                      # recorte del VEHÍCULO (vista del frame)
        self.vehicle_class = vehicle_class    # str: "car", "motorcycle", ...
        self.bbox = bbox                      # np.array [x1, y1, x2, y2] para evidencia
