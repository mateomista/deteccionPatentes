import cv2
import numpy as np
import pytest

from alpr import contracts
from alpr.contracts import (parse_vehicle_crop_stem, read_image, sanitize_video_stem, vehicle_crop_filename,
                            write_image_atomic)


def test_filename_with_and_without_video():
    assert vehicle_crop_filename(17, "trafico01") == "vehiculo_17__trafico01.jpg"
    assert vehicle_crop_filename(17) == "vehiculo_17.jpg"
    assert vehicle_crop_filename(3, "a", ".png") == "vehiculo_3__a.png"
    with pytest.raises(ValueError):
        vehicle_crop_filename(-1, "a")


def test_parse_roundtrip_and_legacy_format():
    assert parse_vehicle_crop_stem("vehiculo_17__trafico01") == (17, "trafico01")
    assert parse_vehicle_crop_stem("vehiculo_17") == (17, None)             # formato antiguo
    assert parse_vehicle_crop_stem("VEHICULO_5__X") == (5, "X")
    for bad in ("vehiculo_x", "vehiculo_", "auto_1", "vehiculo_1__", "vehiculo_1__a b", "vehiculo_1__a/b"):
        assert parse_vehicle_crop_stem(bad) is None, bad
    stem = vehicle_crop_filename(42, "Exportación de ACC 2026").rsplit(".", 1)[0]
    assert parse_vehicle_crop_stem(stem) == (42, sanitize_video_stem("Exportación de ACC 2026"))


def test_sanitize_makes_names_safe_for_windows():
    s = sanitize_video_stem('Exportación de ACC - 2026-09-10 21.30.05-13')
    assert all(ch.isalnum() or ch in "._-" for ch in s) and " " not in s
    assert sanitize_video_stem("a:b*c?d") == "a-b-c-d"
    assert sanitize_video_stem("///") == "video"          # nunca queda vacío


def test_same_track_id_in_two_videos_does_not_collide():
    assert vehicle_crop_filename(1, "video_a") != vehicle_crop_filename(1, "video_b")


def test_write_is_atomic_and_leaves_no_tmp(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (50, 80, 3), dtype=np.uint8)
    out = write_image_atomic(tmp_path / "sub" / "vehiculo_1__v.jpg", img)
    assert out.exists() and not list(tmp_path.rglob("*.tmp"))
    back = cv2.imdecode(np.fromfile(str(out), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert back.shape == img.shape
    write_image_atomic(out, img)                           # sobrescribir no falla
    assert out.exists()


def test_write_never_exposes_a_partial_final_file(tmp_path, monkeypatch):
    """Si falla antes de renombrar, el nombre final no existe y no queda el .tmp."""
    img = np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(contracts.os, "replace", lambda *a, **k: (_ for _ in ()).throw(PermissionError("lock")))
    with pytest.raises(OSError):
        write_image_atomic(tmp_path / "vehiculo_1.jpg", img, retries=2, delay=0.0)
    assert not (tmp_path / "vehiculo_1.jpg").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_retries_transient_lock(tmp_path, monkeypatch):
    img = np.zeros((10, 10, 3), np.uint8)
    real = contracts.os.replace
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("WinError 32")
        return real(a, b)

    monkeypatch.setattr(contracts.os, "replace", flaky)
    assert write_image_atomic(tmp_path / "vehiculo_2.jpg", img, delay=0.0).exists()
    assert calls["n"] == 2


def test_write_rejects_unencodable_image(tmp_path):
    with pytest.raises(Exception):
        write_image_atomic(tmp_path / "vehiculo_1.jpg", None)


def test_read_image_unicode_path(tmp_path):
    img = np.random.default_rng(42).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    sub = tmp_path / "carpeta con ñ y tildes áéíóú"
    path = sub / "imagen_prueba.png"
    write_image_atomic(path, img)

    loaded = read_image(path)
    assert loaded is not None
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


def test_read_image_nonexistent(tmp_path):
    assert read_image(tmp_path / "no_existe_archivo.jpg") is None


def test_read_image_directory_raises_is_a_directory_error(tmp_path):
    d = tmp_path / "directorio_prueba"
    d.mkdir()
    with pytest.raises(IsADirectoryError):
        read_image(d)


def test_read_image_empty_file(tmp_path):
    empty = tmp_path / "vacio.jpg"
    empty.touch()
    assert read_image(empty) is None


def test_read_image_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupto.jpg"
    corrupt.write_bytes(b"esto no es una imagen real")
    assert read_image(corrupt) is None


def test_write_and_read_roundtrip_unicode(tmp_path):
    img = np.random.default_rng(123).integers(0, 255, (50, 50, 3), dtype=np.uint8)
    path = tmp_path / "ruta_con_ñ_y_espacios" / "sub_carpeta" / "foto.jpg"
    write_image_atomic(path, img, jpeg_quality=100)

    loaded = read_image(path)
    assert loaded is not None
    assert loaded.shape == img.shape
    assert loaded.dtype == np.uint8


def test_read_image_propagates_oserror(tmp_path, monkeypatch):
    p = tmp_path / "perm_error.jpg"
    p.touch()
    def fake_fromfile(*args, **kwargs):
        raise PermissionError("Acceso denegado simulado")
    monkeypatch.setattr(np, "fromfile", fake_fromfile)
    with pytest.raises(PermissionError):
        read_image(p)


def test_coco_plate_dataset_reads_unicode_path(tmp_path):
    import json
    from alpr.train.train_rtdetr import CocoPlateDataset

    img_dir = tmp_path / "imágenes con acentos y ñ"
    img_dir.mkdir()
    img_name = "foto_1.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    write_image_atomic(img_dir / img_name, img)

    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps({
        "images": [{"id": 1, "file_name": img_name}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 10, 50, 20], "area": 1000}]
    }), encoding="utf-8")

    def mock_processor(images, annotations, return_tensors):
        return {"pixel_values": [images], "labels": [annotations]}

    ds = CocoPlateDataset(str(ann_file), str(img_dir), mock_processor)
    item = ds[0]
    assert item["pixel_values"].shape == (100, 100, 3)


def test_coco_plate_dataset_raises_on_missing_image(tmp_path):
    import json
    from alpr.train.train_rtdetr import CocoPlateDataset

    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "inexistente.jpg"}],
        "annotations": []
    }), encoding="utf-8")

    ds = CocoPlateDataset(str(ann_file), str(tmp_path), None)
    with pytest.raises(FileNotFoundError):
        _ = ds[0]
