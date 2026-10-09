import cv2
import numpy as np
import pytest

from alpr.tools.cut_video import cut_video, parse_time


def test_parse_time():
    assert parse_time("90") == 90 and parse_time("01:30") == 90 and parse_time("1:00:00.5") == 3600.5
    with pytest.raises(ValueError):
        parse_time("1:2:3:4")


def make(path, n=100, fps=10.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 2, np.uint8))
    w.release()


def test_cut_has_exact_frame_count_and_offset(tmp_path):
    src, out = tmp_path / "s.mp4", tmp_path / "o" / "c.mp4"
    make(src)
    info = cut_video(str(src), str(out), start_s=3.0, duration_s=2.0)   # frames 30..49
    assert info["frames_written"] == 20 and abs(info["fps"] - 10.0) < 0.5
    cap = cv2.VideoCapture(str(out))
    ok, f = cap.read()
    assert ok and abs(int(f.mean()) - 60) < 8      # frame 30 -> brillo ~60


def test_cut_past_end_fails(tmp_path):
    src = tmp_path / "s.mp4"
    make(src, n=20)
    with pytest.raises(RuntimeError):
        cut_video(str(src), str(tmp_path / "c.mp4"), start_s=60.0)
    with pytest.raises(FileNotFoundError):
        cut_video(str(tmp_path / "nope.avi"), str(tmp_path / "c.mp4"))
