"""Humo del pipeline REAL con modelos simulados (sin pesos ni GPU).

Corre ALPRPipeline.run_video sobre un video sintético con un "auto" que cruza la
zona de lectura, con RF-DETR, RT-DETR y EasyOCR reemplazados por dobles. Verifica
la máquina de estados, el reinicio entre videos y, sobre todo, que el REPLAY
offline reproduzca lo que decide la corrida real (la premisa de la calibración).
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("supervision")

from alpr.config import PipelineOptions, Thresholds  # noqa: E402
from alpr.pipeline import ALPRPipeline  # noqa: E402
from alpr.replay import replay_video  # noqa: E402

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


def build(tmp_path, th=None, record=False, **opts):
    models = SimpleNamespace(device="cpu", vehicle_model=FakeVehicleModel(),
                             vehicle_class_names={2: "car"}, ocr_reader=FakeReader(),
                             plate_processor=None, plate_model=None)
    options = PipelineOptions(snapshots_dir=None, strict_plate=True, dual_ocr=False,
                              **opts)
    pipe = ALPRPipeline(models, options, th or Thresholds(), record_observations=record)
    # RT-DETR simulado: una chapa apaisada (aspecto 4) dentro del recorte del vehículo.
    pipe._plate_forward = lambda crops: [np.array([20.0, 70.0, 180.0, 110.0]) for _ in crops]
    return pipe, models


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / "trafico01.mp4"
    make_video(p)
    return str(p)


def key(rec):
    return (rec["patente_validada"], rec["finalization"], rec["frame_idx"],
            rec["cantidad_votos"], tuple(rec["bounding_box"]))


def test_real_run_detects_plate_by_consensus(tmp_path, video):
    pipe, _ = build(tmp_path)
    res = pipe.run_video(video)
    assert [d["patente_validada"] for d in res.detections] == ["AB123CD"]
    assert res.detections[0]["finalization"] == "consensus"
    assert res.detections[0]["snapshot_path"] is None       # snapshots_dir=None
    assert res.audit_problems == [] and res.frames == FRAMES
    assert res.stats["plate_crops_total"] >= 2 and res.stats["ocr_crops_resolved"] >= 2


def test_state_is_reset_between_videos(tmp_path, video):
    pipe, models = build(tmp_path)
    first = pipe.run_video(video)
    models.vehicle_model.i = -1      # el doble de RF-DETR cuenta frames: reiniciarlo
    second = pipe.run_video(video)
    assert [key(d) for d in first.detections] == [key(d) for d in second.detections]
    assert second.stats["records_consensus"] == 1     # no se acumuló del run anterior


@pytest.mark.parametrize("th", [
    Thresholds(),                                                   # consenso
    Thresholds(min_votes=50, early_finalize_votes=50),              # sin consenso -> track_lost
    Thresholds(min_votes=2, min_consensus_margin=0.0, early_finalize_votes=2,
               early_finalize_conf=0.5),                            # vía rápida
])
def test_replay_reproduces_the_real_run(tmp_path, video, th):
    real, _ = build(tmp_path, th=th)
    real_res = real.run_video(video)

    rec, _ = build(tmp_path, th=th, record=True)
    obs = rec.run_video(video).observations
    assert obs["tracks"] and obs["total_frames"] == FRAMES
    replayed = replay_video(obs, th, flush_unfinalized=True, dedup_seconds=30.0)

    assert [key(r) for r in replayed] == [key(r) for r in real_res.detections]
    assert len(real_res.detections) == 1


def test_no_consensus_run_is_recovered_as_track_lost(tmp_path, video):
    th = Thresholds(min_votes=50, early_finalize_votes=50)
    pipe, _ = build(tmp_path, th=th)
    res = pipe.run_video(video)
    assert [d["finalization"] for d in res.detections] == ["track_lost"]
    pipe2, _ = build(tmp_path, th=th, flush_unfinalized=False)
    assert pipe2.run_video(video).detections == []


def test_record_mode_never_finalizes_and_stores_no_images(tmp_path, video):
    rec, _ = build(tmp_path, record=True)
    res = rec.run_video(video)
    assert res.detections == [] and res.observations["strict_plate"] is True
    votes = next(iter(res.observations["tracks"].values()))["votes"]
    assert len(votes) > 5 and all(set(v) == {"frame", "text", "conf", "bbox"} for v in votes)
