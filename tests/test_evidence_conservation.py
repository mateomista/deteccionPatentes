"""Pruebas de la Fase 2: Conservacion de evidencia y trazabilidad de descartes (H-1, H-2, H-4)."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("supervision")

from alpr.config import MIN_VEHICLE_CROP_HEIGHT, MIN_VEHICLE_CROP_WIDTH, PipelineOptions, Thresholds
from alpr.contracts import read_image
from alpr.pipeline import ALPRPipeline
from alpr.tracking import TrackState, VehicleTrack

W, H, FRAMES, VEH_H, VEH_W = 640, 480, 300, 120, 200


def make_video(path):
    rng = np.random.default_rng(0)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert out.isOpened()
    for i in range(FRAMES):
        frame = np.full((H, W, 3), 90, np.uint8)
        y = 4 * i
        if y < H:   # el auto sale del cuadro hacia el frame ~120 y no vuelve
            frame[y:min(H, y + VEH_H), 200:200 + VEH_W] = rng.integers(
                0, 255, (min(H, y + VEH_H) - y, VEH_W, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


class FakeVehicleModel:
    def __init__(self):
        self.i = -1

    def predict(self, frame_rgb, threshold):
        self.i += 1
        y = 4 * self.i
        if y >= H:
            return SimpleNamespace(xyxy=np.zeros((0, 4)), confidence=np.zeros(0), class_id=np.zeros(0))
        return SimpleNamespace(xyxy=np.array([[200, y, 200 + VEH_W, min(H, y + VEH_H)]], float),
                               confidence=np.array([0.9]), class_id=np.array([2]))


class FakeReader:
    def readtext(self, image, **kw):
        return [([[0, 0], [100, 0], [100, 30], [0, 30]], "AB123CD", 0.9)]


class BadReader:
    """EasyOCR que solo lee basura que no es una patente."""
    def readtext(self, image, **kw):
        return [([[0, 0], [100, 0], [100, 30], [0, 30]], "XXXX", 0.8)]


def build(tmp_path, th=None, record=False, reader=None, snapshots_dir=None, **opts):
    models = SimpleNamespace(device="cpu", vehicle_model=FakeVehicleModel(),
                             vehicle_class_names={2: "car"}, ocr_reader=reader or FakeReader(),
                             plate_processor=None, plate_model=None)
    options = PipelineOptions(snapshots_dir=snapshots_dir, strict_plate=True, dual_ocr=False,
                              **opts)
    pipe = ALPRPipeline(models, options, th or Thresholds(), record_observations=record)
    # RT-DETR simulado: una chapa apaisada (aspecto 4) dentro del recorte del vehiculo.
    pipe._plate_forward = lambda crops: [np.array([20.0, 70.0, 180.0, 110.0]) for _ in crops]
    return pipe, models


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / "trafico01.mp4"
    make_video(p)
    return str(p)


# ------------------------------------------------------------------------------
# H-1: Seleccion determinista de la evidencia representativa
# ------------------------------------------------------------------------------
def test_deterministic_selection_of_representative_vehicle_evidence():
    t = VehicleTrack(1, "car")
    assert t.best_vehicle_crop is None
    assert t.best_vehicle_area == 0

    # 1. Primer recorte: 100x80 (area 8.000) en frame 10
    crop1 = np.full((80, 100, 3), 10, dtype=np.uint8)
    t.update_vehicle_evidence(crop1, 10, (0, 0, 100, 80))
    assert t.best_vehicle_area == 8000
    assert t.best_vehicle_frame == 10
    assert np.array_equal(t.best_vehicle_crop, crop1)

    # 2. Recorte mas chico: 50x50 (area 2.500) en frame 15 -> no debe reemplazar
    crop2 = np.full((50, 50, 3), 20, dtype=np.uint8)
    t.update_vehicle_evidence(crop2, 15, (0, 0, 50, 50))
    assert t.best_vehicle_area == 8000
    assert t.best_vehicle_frame == 10
    assert np.array_equal(t.best_vehicle_crop, crop1)

    # 3. Recorte mas grande: 160x120 (area 19.200) en frame 20 -> debe reemplazar
    crop3 = np.full((120, 160, 3), 30, dtype=np.uint8)
    t.update_vehicle_evidence(crop3, 20, (0, 0, 160, 120))
    assert t.best_vehicle_area == 19200
    assert t.best_vehicle_frame == 20
    assert np.array_equal(t.best_vehicle_crop, crop3)

    # 4. Desempate por fotograma mas reciente ante areas iguales: area 19.200 en frame 25
    crop4 = np.full((120, 160, 3), 40, dtype=np.uint8)
    t.update_vehicle_evidence(crop4, 25, (10, 10, 170, 130))
    assert t.best_vehicle_area == 19200
    assert t.best_vehicle_frame == 25
    assert np.array_equal(t.best_vehicle_crop, crop4)


# ------------------------------------------------------------------------------
# H-1: Evidencia conservada cuando OCR no devuelve lectura
# ------------------------------------------------------------------------------
def test_vehicle_evidence_preserved_when_ocr_fails(tmp_path, video):
    snaps = tmp_path / "snaps"
    pipe, _ = build(tmp_path, reader=BadReader(), snapshots_dir=str(snaps))
    res = pipe.run_video(video)

    assert len(res.detections) == 0
    # Descarte a nivel vehiculo
    veh_disc = next(d for d in res.discards if d["motivo"] == "vehiculo_sin_lecturas_validas")
    assert veh_disc["recorte_vehiculo"] is not None
    veh_path = Path(veh_disc["recorte_vehiculo"])
    assert veh_path.exists()
    assert "vehiculos" in str(veh_path)
    loaded = read_image(veh_path)
    assert loaded is not None
    assert loaded.shape[1] >= MIN_VEHICLE_CROP_WIDTH and loaded.shape[0] >= MIN_VEHICLE_CROP_HEIGHT


# ------------------------------------------------------------------------------
# H-1: Evidencia conservada cuando RT-DETR no encuentra patente
# ------------------------------------------------------------------------------
def test_vehicle_evidence_preserved_when_rtdetr_finds_no_plate(tmp_path, video):
    snaps = tmp_path / "snaps"
    pipe, _ = build(tmp_path, snapshots_dir=str(snaps))
    # Simulamos que RT-DETR devuelve None para todos los recortes
    pipe._plate_forward = lambda crops: [None for _ in crops]
    res = pipe.run_video(video)

    assert len(res.detections) == 0
    assert any(d["motivo"] == "rtdetr_sin_patente" for d in res.discards)
    veh_disc = next(d for d in res.discards if d["motivo"] == "vehiculo_sin_lecturas_validas")
    assert veh_disc["recorte_vehiculo"] is not None
    assert Path(veh_disc["recorte_vehiculo"]).exists()


# ------------------------------------------------------------------------------
# H-1: Evidencia conservada para seguimiento que termina sin consenso
# ------------------------------------------------------------------------------
def test_vehicle_evidence_preserved_when_track_ends_without_consensus(tmp_path, video):
    snaps = tmp_path / "snaps"
    # Umbrales inalcanzables para consenso y sin recuperacion (votos con conf 0.90 < 0.95)
    th = Thresholds(min_votes=50, early_finalize_votes=50, flush_single_vote_conf=0.95)
    pipe, _ = build(tmp_path, th=th, snapshots_dir=str(snaps))
    res = pipe.run_video(video)

    assert len(res.detections) == 0
    veh_disc = next(d for d in res.discards if d["motivo"] == "vehiculo_sin_consenso_no_recuperable")
    assert "lecturas" in veh_disc and "AB123CD" in veh_disc["lecturas"]
    assert veh_disc["recorte_vehiculo"] is not None
    assert Path(veh_disc["recorte_vehiculo"]).exists()


# ------------------------------------------------------------------------------
# H-1: Evidencia conservada cuando la patente si se registra correctamente
# ------------------------------------------------------------------------------
def test_vehicle_evidence_preserved_when_plate_registered(tmp_path, video):
    snaps = tmp_path / "snaps"
    pipe, _ = build(tmp_path, snapshots_dir=str(snaps))
    res = pipe.run_video(video)

    assert len(res.detections) == 1
    det = res.detections[0]
    assert det["snapshot_path"] is not None
    assert det.get("snapshot_vehiculo_path") is not None
    plate_path = Path(det["snapshot_path"])
    veh_path = Path(det["snapshot_vehiculo_path"])
    assert plate_path.exists()
    assert veh_path.exists()
    assert plate_path != veh_path


# ------------------------------------------------------------------------------
# H-1: Fallo de escritura atomica sin caida ni referencia falsa
# ------------------------------------------------------------------------------
def test_vehicle_evidence_write_failure_does_not_crash_or_fabricate_path(tmp_path, video, monkeypatch):
    import alpr.pipeline as pipeline_mod
    snaps = tmp_path / "snaps"
    pipe, _ = build(tmp_path, snapshots_dir=str(snaps))

    real_write = pipeline_mod.write_image_atomic

    def flaky_write(path, img, **kwargs):
        if "vehiculos" in str(path):
            raise OSError("Fallo simulado de disco lleno al escribir vehiculo")
        return real_write(path, img, **kwargs)

    monkeypatch.setattr(pipeline_mod, "write_image_atomic", flaky_write)
    res = pipe.run_video(video)

    # La deteccion se completa exitosamente
    assert len(res.detections) == 1
    det = res.detections[0]
    assert det["snapshot_path"] is not None
    # No inventa una ruta de archivo existente para el vehiculo
    assert det.get("snapshot_vehiculo_path") is None


# ------------------------------------------------------------------------------
# H-2: Descarte por perdida del seguimiento con motivo final explicito
# ------------------------------------------------------------------------------
def test_discard_on_track_lost_has_explicit_final_reason(tmp_path, video):
    snaps = tmp_path / "snaps"
    pipe, _ = build(tmp_path, reader=BadReader(), snapshots_dir=str(snaps))
    res = pipe.run_video(video)

    disc = next(d for d in res.discards if d["motivo"] == "vehiculo_sin_lecturas_validas")
    assert disc["id_vehiculo"] is not None
    assert disc["tipo_vehiculo"] == "car"
    assert disc["ultimo_fotograma"] is not None and disc["ultimo_fotograma"] >= 0
    assert disc["frame_idx"] == disc["ultimo_fotograma"]


# ------------------------------------------------------------------------------
# H-2: Seguimiento registrado correctamente no produce descarte final falso
# ------------------------------------------------------------------------------
def test_registered_vehicle_has_no_false_final_discard(tmp_path, video):
    pipe, _ = build(tmp_path)
    res = pipe.run_video(video)
    assert len(res.detections) == 1
    registered_id = res.detections[0]["id_vehiculo"]

    # Ningun descarte a nivel vehiculo para el ID registrado
    false_discards = [d for d in res.discards if d["id_vehiculo"] == registered_id
                      and d["motivo"].startswith("vehiculo_")]
    assert false_discards == []


# ------------------------------------------------------------------------------
# H-2: log_discards=False sin registros de descartes
# ------------------------------------------------------------------------------
def test_log_discards_false_disables_discard_records(tmp_path, video):
    pipe, _ = build(tmp_path, reader=BadReader(), log_discards=False)
    res = pipe.run_video(video)
    assert res.discards == []


# ------------------------------------------------------------------------------
# H-4: Recorte inferior a dimensiones minimas no se encola y queda trazado
# ------------------------------------------------------------------------------
class SmallVehicleModel:
    def __init__(self):
        self.i = -1

    def predict(self, frame_rgb, threshold):
        self.i += 1
        # Vehiculo muy chico: 40x30 pixeles (menor a 64x48)
        y = 200 + 2 * self.i
        if y >= 400:
            return SimpleNamespace(xyxy=np.zeros((0, 4)), confidence=np.zeros(0), class_id=np.zeros(0))
        return SimpleNamespace(xyxy=np.array([[200, y, 240, y + 30]], float),
                               confidence=np.array([0.9]), class_id=np.array([2]))


def test_crop_below_minimum_dimensions_is_logged_with_dimensions_and_not_enqueued(tmp_path, video):
    models = SimpleNamespace(device="cpu", vehicle_model=SmallVehicleModel(),
                             vehicle_class_names={2: "car"}, ocr_reader=FakeReader(),
                             plate_processor=None, plate_model=None)
    options = PipelineOptions(snapshots_dir=str(tmp_path / "snaps"), strict_plate=True, dual_ocr=False)
    pipe = ALPRPipeline(models, options, Thresholds())
    forward_called = {"count": 0}

    def fake_plate_forward(crops):
        forward_called["count"] += len(crops)
        return [np.array([10, 10, 30, 20]) for _ in crops]

    pipe._plate_forward = fake_plate_forward
    res = pipe.run_video(video)

    assert forward_called["count"] == 0  # No se encolo ningun forward
    assert res.stats["vehicle_crops_skipped"] > 0
    assert res.stats["vehicle_crops_too_small"] > 0

    small_disc = [d for d in res.discards if d["motivo"] == "recorte_vehiculo_demasiado_pequeno"]
    assert len(small_disc) == 1  # Control de repeticion: exactamente 1 registro para el track
    sd = small_disc[0]
    assert sd["ancho"] == 40 and sd["alto"] == 30
    assert sd["min_ancho"] == MIN_VEHICLE_CROP_WIDTH and sd["min_alto"] == MIN_VEHICLE_CROP_HEIGHT
    assert sd["tipo_vehiculo"] == "car"


# ------------------------------------------------------------------------------
# H-1 / H-2: Ausencia de duplicados innecesarios en registro y evidencia
# ------------------------------------------------------------------------------
def test_no_duplicate_discard_or_evidence_for_same_track(tmp_path, video):
    snaps = tmp_path / "snaps"
    pipe, _ = build(tmp_path, reader=BadReader(), snapshots_dir=str(snaps))
    res = pipe.run_video(video)

    # Cada track_id tiene como maximo UN descarte final de nivel vehiculo
    track_level_discards = [d for d in res.discards if d["motivo"].startswith("vehiculo_")]
    track_ids = [d["id_vehiculo"] for d in track_level_discards]
    assert len(track_ids) == len(set(track_ids))

    # Las imagenes en vehiculos/ no estan duplicadas
    veh_dir = snaps / "vehiculos"
    if veh_dir.exists():
        files = list(veh_dir.glob("*.jpg"))
        names = [f.name for f in files]
        assert len(names) == len(set(names))
