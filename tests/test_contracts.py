import cv2
import numpy as np
import pytest

from alpr import contracts
from alpr.contracts import (parse_vehicle_crop_stem, sanitize_video_stem, vehicle_crop_filename,
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
