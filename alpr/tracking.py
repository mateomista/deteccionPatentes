"""Estado por vehículo: votos, evidencia y criterio de consenso."""
from collections import defaultdict
from enum import Enum

from .plate_text import fuse_plate_reads, matches_official_format, plate_distance

# Voto por caracteres: una lectura respalda al texto fusionado si difiere en a lo sumo este número
# de posiciones (con el mismo largo), y hacen falta al menos FUSION_MIN_READS lecturas así: con
# dos lecturas distintas "fusionar" es elegir la más confiable, y no es un consenso.
FUSION_MAX_DISTANCE = 2
FUSION_MIN_READS = 3


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
        self.vehicle_crop = None    # recorte del VEHÍCULO completo (solo si se exporta)


class VehicleTrack:
    """Centraliza todo el estado y la evidencia asociada a un track_id."""
    def __init__(self, track_id, vehicle_class_name, char_voting=False):
        self.track_id = track_id
        self.char_voting = bool(char_voting)
        self.vehicle_class_name = vehicle_class_name
        self.state = TrackState.NEW
        self.last_seen_frame = -1
        self.last_ocr_frame = -999

        self.votes = {}        # {plate_text: sum_confidence}
        self.vote_counts = {}  # {plate_text: count}
        self.evidence = defaultdict(PlateEvidence)

        self.best_plate_text = None

    def add_vote(self, text, confidence, frame_idx, plate_crop, processed_crop, vehicle_bbox,
                 vehicle_crop=None):
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
            ev.vehicle_crop = vehicle_crop.copy() if vehicle_crop is not None else None

    def get_consensus_plate(self, fuse=None):
        """Patente ganadora: por sumatoria de confianza del texto entero o, con `char_voting`, por
        voto carácter por carácter entre las lecturas del track. Devuelve (texto, confianza media,
        cantidad de lecturas que la respaldan). `fuse` fuerza (True) o impide (False) el voto por
        caracteres; por defecto lo decide `char_voting`."""
        return self._consensus(self.char_voting if fuse is None else fuse)[:3]

    def _consensus(self, fuse):
        """(texto, confianza media, lecturas, fusionado): `fusionado` es True si el ganador salió del
        voto por caracteres."""
        if not self.votes:
            return None, 0.0, 0, False
        if fuse and len(self.votes) > 1:
            fused = self._fused_consensus()
            if fused is not None:
                return (*fused, True)
        best_plate = max(self.votes, key=self.votes.get)
        count = self.vote_counts[best_plate]
        return best_plate, self.votes[best_plate] / count, count, False

    def _fused_consensus(self):
        """Consenso por caracteres, o None si no corresponde (se usa el voto por texto entero).

        No corresponde si el texto fusionado rompe el formato oficial que sí cumple el ganador por
        texto entero: un texto armado de partes no debe ser peor que una lectura real."""
        fused = fuse_plate_reads(self.votes)
        by_text = max(self.votes, key=self.votes.get)
        if fused is None or (matches_official_format(by_text) and not matches_official_format(fused)):
            return None
        near = [t for t in self.votes
                if len(t) == len(fused) and plate_distance(t, fused) <= FUSION_MAX_DISTANCE]
        if not near:
            return None
        count = sum(self.vote_counts[t] for t in near)
        if count < FUSION_MIN_READS:
            return None
        avg_conf = sum(self.votes[t] for t in near) / count
        if fused not in self.votes:
            # La evidencia de un texto armado es la de su lectura más parecida y más confiable.
            src = min(near, key=lambda t: (plate_distance(t, fused), -self.evidence[t].best_conf))
            ev = PlateEvidence()
            ev.__dict__.update(self.evidence[src].__dict__)
            ev.text = fused
            self.evidence[fused] = ev
        return fused, avg_conf, count

    def check_finalization(self, th):
        """`th`: Thresholds. Los umbrales llegan por parámetro (no globales).

        Finaliza por consenso solo con lecturas IDÉNTICAS: el voto por caracteres no adelanta la
        decisión (con pocas lecturas puede elegir mal), solo participa cuando el track se recupera
        sin consenso (`flush_eligible`)."""
        if self.state == TrackState.FINALIZED:
            return True
        best_plate, avg_conf, count, fused = self._consensus(False)
        if not best_plate:
            return False

        # El margen compara textos enteros: con el voto por caracteres no aplica.
        margin = 1.0
        if not fused and len(self.votes) > 1:
            best_score = self.votes[best_plate]
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
