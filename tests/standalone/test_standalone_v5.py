"""Pruebas de standalone/pipelineV5.py (V5 con exportación de recortes de vehículo).

Verifican, sobre un video sintético y con los modelos simulados:
  - la exportación escribe lo que dice el contrato;
  - con la exportación APAGADA el resultado es idéntico al V5 original (legacy/);
  - V5 con mejoras y el pipeline modular producen lo mismo;
  - el contrato (nombres) no diverge entre las copias.
"""
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("supervision")
pytest.importorskip("transformers")

from standalone_loader import load, load_standalone_classifier  # noqa: E402

W, H, FRAMES, VEH_W, VEH_H = 640, 480, 300, 200, 120
VIDEO_NAME = "trafico01"


def make_video(path):
    rng = np.random.default_rng(0)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for i in range(FRAMES):
        frame = np.full((H, W, 3), 90, np.uint8)
        y = 4 * i
        if y < H:
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


PLATE_BOX = np.array([20.0, 70.0, 180.0, 110.0])


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / f"{VIDEO_NAME}.mp4"
    make_video(p)
    return str(p)


@pytest.fixture(scope="module")
def V5():
    mod = load("standalone_v5", "standalone/pipelineV5.py")
    mod.VEHICLE_CLASS_NAMES = {2: "car"}          # el doble de RF-DETR devuelve class_id 2
    return mod


@pytest.fixture(scope="module")
def LEGACY():
    mod = load("legacy_v5", "legacy/pipelineV5.py")
    mod.VEHICLE_CLASS_NAMES = {2: "car"}
    return mod


def run_v5(mod, monkeypatch, tmp_path, video, tag, export_dir=None):
    """Corre un V5 (original o con mejoras) con modelos simulados. Devuelve (detecciones, pipeline)."""
    model_dir = tmp_path / "plate_model"
    model_dir.mkdir(exist_ok=True)
    out_json = tmp_path / f"{tag}.json"
    argv = ["pipelineV5.py", "--source", video, "--plate-model", str(model_dir),
            "--snapshots-dir", str(tmp_path / f"snaps_{tag}"), "--output-json", str(out_json),
            "--output-video", "", "--strict-plate", "--no-dual-ocr"]
    if export_dir:
        argv += ["--export-vehicle-crops", str(export_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    args = mod.parse_args()

    def fake_load(self):
        self.vehicle_model = FakeVehicleModel()
        self.plate_processor = None
        self.plate_model = None
        self.ocr_reader = FakeReader()

    monkeypatch.setattr(mod.ALPRPipeline, "_load_models", fake_load)
    pipe = mod.ALPRPipeline(args)
    pipe._detect_plate_box_batch = lambda crops: [PLATE_BOX for _ in crops]
    pipe.run()
    return json.loads(out_json.read_text(encoding="utf-8")), pipe


def key(rec):
    return (rec["patente_validada"], rec["finalization"], rec["frame_idx"], rec["cantidad_votos"],
            tuple(rec["bounding_box"]), rec["confianza_ocr_promedio"], rec["tipo_vehiculo"])


def test_export_writes_the_contract_file(V5, monkeypatch, tmp_path, video):
    export = tmp_path / "crops"
    dets, pipe = run_v5(V5, monkeypatch, tmp_path, video, "on", export)
    assert len(dets) == 1
    tid = dets[0]["id_vehiculo"]
    assert sorted(p.name for p in export.iterdir()) == [f"vehiculo_{tid}__{VIDEO_NAME}.jpg"]
    img = cv2.imdecode(np.fromfile(str(next(export.iterdir())), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[1] >= 64 and img.shape[0] >= 48
    assert pipe.stats["vehicle_crops_exported"] == 1 and not list(export.glob("*.tmp"))


def test_export_is_off_by_default(V5, monkeypatch, tmp_path, video):
    _, pipe = run_v5(V5, monkeypatch, tmp_path, video, "off")
    assert pipe.export_vehicle_dir is None and pipe.stats["vehicle_crops_exported"] == 0


def test_with_export_off_the_result_is_identical_to_the_original_v5(V5, LEGACY, monkeypatch, tmp_path, video):
    mejorado, _ = run_v5(V5, monkeypatch, tmp_path, video, "mejorado")
    original, _ = run_v5(LEGACY, monkeypatch, tmp_path, video, "original")
    assert len(original) == 1
    assert [key(d) for d in mejorado] == [key(d) for d in original]


def test_export_on_or_off_gives_the_same_detections(V5, monkeypatch, tmp_path, video):
    off, _ = run_v5(V5, monkeypatch, tmp_path, video, "a")
    on, _ = run_v5(V5, monkeypatch, tmp_path, video, "b", tmp_path / "crops_b")
    assert [key(d) for d in off] == [key(d) for d in on]


def test_standalone_v5_and_modular_pipeline_agree(V5, monkeypatch, tmp_path, video):
    """Las mismas mejoras en las dos versiones: mismas detecciones y mismo archivo exportado."""
    from alpr.config import PipelineOptions, Thresholds
    from alpr.pipeline import ALPRPipeline

    v5_dets, _ = run_v5(V5, monkeypatch, tmp_path, video, "v5", tmp_path / "crops_v5")

    models = SimpleNamespace(device="cpu", vehicle_model=FakeVehicleModel(), vehicle_class_names={2: "car"},
                             ocr_reader=FakeReader(), plate_processor=None, plate_model=None)
    options = PipelineOptions(snapshots_dir=None, strict_plate=True, dual_ocr=False,
                              export_vehicle_dir=str(tmp_path / "crops_mod"))
    pipe = ALPRPipeline(models, options, Thresholds())
    pipe._plate_forward = lambda crops: [PLATE_BOX for _ in crops]
    mod_dets = pipe.run_video(video).detections

    strip = lambda d: key(d)  # noqa: E731
    assert [strip(d) for d in v5_dets] == [strip(d) for d in mod_dets]
    assert sorted(p.name for p in (tmp_path / "crops_v5").iterdir()) == \
           sorted(p.name for p in (tmp_path / "crops_mod").iterdir())


# ------------------------------------------------------------- el contrato no diverge entre copias
@pytest.mark.parametrize("track_id,video", [(1, "a"), (17, "trafico01"), (3, "Exportación de ACC - 2026"),
                                           (5, "a:b*c"), (9, "///"), (2, None)])
def test_contract_is_identical_in_every_copy(V5, track_id, video):
    from alpr import contracts
    from alpr.tools import join_results
    assert V5.vehicle_crop_filename(track_id, video) == contracts.vehicle_crop_filename(track_id, video)
    if video:
        assert V5.sanitize_video_stem(video) == contracts.sanitize_video_stem(video) \
               == join_results.sanitize_video_stem(video)


def test_standalone_classifier_reads_what_standalone_v5_writes(V5):
    from pathlib import Path
    C = load_standalone_classifier()
    name = V5.vehicle_crop_filename(17, "Exportación de ACC")
    stem = Path(name).stem
    assert C.TRACK_PATTERN.match(stem)
    assert int(C.TRACK_PATTERN.match(stem).group(1)) == 17
    assert C.extract_video(Path(name)) == V5.sanitize_video_stem("Exportación de ACC")


def test_standalone_join_results_is_the_same_tool_as_the_package_one():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    assert (root / "standalone" / "join_results.py").read_text(encoding="utf-8") == \
           (root / "alpr" / "tools" / "join_results.py").read_text(encoding="utf-8")
