import json

import pytest

from alpr.calibrate.calibrate import DEFAULT_GRID, iter_grid, main as calibrate_main, score_row

BBOX = [0, 0, 10, 10]


def make_obs(path, source, plate, confs, with_noise=False):
    tracks = {"1": {"vehicle_class": "car", "last_seen_frame": 300,
                    "votes": [{"frame": 100 + 5 * i, "text": plate, "conf": c, "bbox": BBOX}
                              for i, c in enumerate(confs)]}}
    if with_noise:   # un track espurio de una sola lectura de baja confianza
        tracks["2"] = {"vehicle_class": "car", "last_seen_frame": 600,
                       "votes": [{"frame": 500, "text": "ZZ999ZZ", "conf": 0.55, "bbox": BBOX}]}
    obs = {"source": source, "fps": 30.0, "total_frames": 2000, "strict_plate": True,
           "detection_thresholds": {"confidence_vehicle": 0.45, "confidence_plate": 0.25},
           "tracks": tracks}
    path.write_text(json.dumps(obs))


def test_iter_grid_rejects_detection_fields():
    with pytest.raises(SystemExit):
        list(iter_grid({"confidence_plate": [0.2]}))
    assert len(list(iter_grid({"min_votes": [1, 2], "min_avg_ocr_conf": [0.1, 0.2, 0.3]}))) == 6


def test_score_row_enforces_precision_floor():
    hi_p = {"precision": 0.97, "recall": 0.5, "f1": 0.66}
    hi_r = {"precision": 0.80, "recall": 0.9, "f1": 0.85}
    key = lambda r: score_row(r, "recall_at_precision", 0.95)
    assert key(hi_p) > key(hi_r)
    assert score_row(hi_r, "f1", 0.95) > score_row(hi_p, "f1", 0.95)


def test_sweep_end_to_end_picks_thresholds_that_reject_the_spurious_track(tmp_path):
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    make_obs(obs_dir / "a__cv0.45_cp0.25.json", "a.mp4", "AB123CD", [0.8, 0.8, 0.9], with_noise=True)
    make_obs(obs_dir / "b__cv0.45_cp0.25.json", "b.mp4", "EF456GH", [0.7, 0.6])
    gt = tmp_path / "gt.csv"
    gt.write_text("video,plate,first_frame,last_frame\na,AB123CD,90,200\nb,EF456GH,90,200\n")
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"min_votes": [1, 2], "flush_single_vote_conf": [0.4, 0.9]}))
    out = tmp_path / "out"
    calibrate_main(["sweep", "--obs-dir", str(obs_dir), "--ground-truth", str(gt),
                    "--out-dir", str(out), "--grid", str(grid), "--min-precision", "1.0"])

    best = json.loads((out / "best_thresholds.json").read_text())
    assert best["objective_reached"] is True
    # Con min_votes=1 el track espurio (1 lectura, 0.55) finalizaría: la grilla debe evitarlo.
    assert best["thresholds"]["min_votes"] == 2
    assert best["metrics_on_calibration_set"]["precision"] == 1.0
    assert best["metrics_on_calibration_set"]["recall"] == 1.0
    assert (out / "sweep_results.csv").exists()
    # best_thresholds.json es directamente consumible por Thresholds.from_json
    from alpr.config import Thresholds
    assert Thresholds.from_json(out / "best_thresholds.json").min_votes == 2


def test_sweep_refuses_obs_without_gt(tmp_path):
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    make_obs(obs_dir / "x__cv0.45_cp0.25.json", "x.mp4", "AB123CD", [0.8, 0.8])
    gt = tmp_path / "gt.csv"
    gt.write_text("video,plate,first_frame,last_frame\nother,AB123CD,0,10\n")
    with pytest.raises(SystemExit):
        calibrate_main(["sweep", "--obs-dir", str(obs_dir), "--ground-truth", str(gt),
                        "--out-dir", str(tmp_path / "o")])


def test_default_grid_only_has_decision_fields():
    combos = list(iter_grid(DEFAULT_GRID))
    assert len(combos) == 3 * 5 * 3 * 3 * 3 * 4
