import numpy as np
import pytest

from alpr.config import Thresholds
from alpr.replay import replay_video
from alpr.tracking import TrackState, VehicleTrack

BBOX = [10, 10, 200, 120]


def make_track(votes, vclass="car"):
    t = VehicleTrack(1, vclass)
    t.state = TrackState.OCR_ACTIVE
    for i, (text, conf) in enumerate(votes):
        t.add_vote(text, conf, i * 5, np.ones((4, 4), np.uint8), None, np.array(BBOX))
    return t


def test_standard_consensus_needs_votes_conf_and_margin():
    th = Thresholds()
    assert not make_track([("AB123CD", 0.8)]).check_finalization(th)
    assert make_track([("AB123CD", 0.8), ("AB123CD", 0.7)]).check_finalization(th)
    # Competidor casi empatado: margen insuficiente.
    t = make_track([("AB123CD", 0.8), ("AB123CD", 0.8), ("XY999ZZ", 0.8), ("XY999ZZ", 0.7)])
    assert not t.check_finalization(th)


def test_early_path_skips_margin():
    th = Thresholds()
    votes = [("AB123CD", 0.95)] * 3 + [("XY999ZZ", 0.95)] * 2
    t = make_track(votes[:3])
    assert t.check_finalization(th)


def test_flush_eligibility():
    th = Thresholds()
    assert make_track([("AB123CD", 0.6)]).flush_eligible(th)             # 1 voto, oficial, conf>=0.5
    assert not make_track([("AB123CD", 0.3)]).flush_eligible(th)        # conf baja
    assert not make_track([("ABCDEFG", 0.9)]).flush_eligible(th)        # 1 voto, no oficial
    assert make_track([("ABCDEFG", 0.9)] * 2).flush_eligible(th)        # 2 votos: se acepta


def obs(tracks, fps=30.0, total=1000, strict=True):
    return {"source": "v.mp4", "fps": fps, "total_frames": total, "strict_plate": strict,
            "tracks": tracks}


def votes(plate, confs, start=100, step=5):
    return [{"frame": start + i * step, "text": plate, "conf": c, "bbox": BBOX}
            for i, c in enumerate(confs)]


def test_replay_consensus_record():
    o = obs({"1": {"vehicle_class": "car", "last_seen_frame": 300,
                   "votes": votes("AB123CD", [0.8, 0.7, 0.9])}})
    recs = replay_video(o, Thresholds())
    assert len(recs) == 1
    r = recs[0]
    assert r["patente_validada"] == "AB123CD" and r["finalization"] == "consensus"
    assert r["cantidad_votos"] == 2          # finaliza al 2º voto (min_votes=2)
    assert r["frame_idx"] == 100             # evidencia = el voto de mayor confianza (0.8)


def test_replay_threshold_changes_decision():
    tr = {"1": {"vehicle_class": "car", "last_seen_frame": 300, "votes": votes("AB123CD", [0.8, 0.7])}}
    assert len(replay_video(obs(tr), Thresholds(min_votes=2))) == 1
    # Con 3 votos exigidos no hay consenso; la recuperación lo rescata como track_lost.
    recs = replay_video(obs(tr), Thresholds(min_votes=3))
    assert [r["finalization"] for r in recs] == ["track_lost"]
    assert replay_video(obs(tr), Thresholds(min_votes=3), flush_unfinalized=False) == []


def test_replay_candidate_floor_filters_votes():
    tr = {"1": {"vehicle_class": "car", "last_seen_frame": 300, "votes": votes("AB123CD", [0.15, 0.15])}}
    assert replay_video(obs(tr), Thresholds(min_avg_ocr_conf=0.2)) == []
    assert len(replay_video(obs(tr), Thresholds(min_avg_ocr_conf=0.1))) == 1


def test_replay_end_of_video_vs_track_lost():
    tr = {"1": {"vehicle_class": "car", "last_seen_frame": 990, "votes": votes("AB123CD", [0.9])}}
    recs = replay_video(obs(tr, total=1000), Thresholds())
    assert recs[0]["finalization"] == "end_of_video"
    tr["1"]["last_seen_frame"] = 200
    recs = replay_video(obs(tr, total=1000), Thresholds())
    assert recs[0]["finalization"] == "track_lost"


def test_replay_dedups_same_plate_across_tracks():
    tr = {"1": {"vehicle_class": "car", "last_seen_frame": 300, "votes": votes("AB123CD", [0.8, 0.8])},
          "2": {"vehicle_class": "car", "last_seen_frame": 400, "votes": votes("AB123CD", [0.8, 0.8], start=300)}}
    assert len(replay_video(obs(tr), Thresholds(), dedup_seconds=30.0)) == 1
    assert len(replay_video(obs(tr), Thresholds(), dedup_seconds=0)) == 2


def test_replay_rejects_invalid_strict_plate():
    tr = {"1": {"vehicle_class": "car", "last_seen_frame": 300, "votes": votes("ABCDEFG", [0.9, 0.9])}}
    assert replay_video(obs(tr, strict=True), Thresholds()) == []
    assert len(replay_video(obs(tr, strict=False), Thresholds())) == 1


def test_thresholds_validation_and_json(tmp_path):
    with pytest.raises(ValueError):
        Thresholds(min_votes=0)
    with pytest.raises(ValueError):
        Thresholds(confidence_plate=1.5)
    p = tmp_path / "t.json"
    th = Thresholds(min_votes=3, min_consensus_margin=0.3)
    th.to_json(p)
    assert Thresholds.from_json(p) == th
    with pytest.raises(ValueError):
        Thresholds.from_dict({"nope": 1})
