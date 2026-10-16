import pytest

from alpr.tools.vehicle_crop_report import analyze_observations, format_report, touches_edge

W, H = 1920, 1080


def test_touches_edge_each_side_and_margin():
    assert touches_edge([1, 100, 800, 700], (W, H))                 # izquierdo
    assert touches_edge([100, 0, 800, 700], (W, H))                 # superior
    assert touches_edge([100, 100, 1917, 700], (W, H))              # derecho
    assert touches_edge([100, 100, 800, 1078], (W, H))              # inferior
    assert not touches_edge([100, 100, 800, 700], (W, H))
    assert not touches_edge([6, 100, 800, 700], (W, H), margin=5)   # justo fuera del margen
    assert touches_edge([6, 100, 800, 700], (W, H), margin=10)


def vote(frame, conf, bbox, text="AB123CD"):
    return {"frame": frame, "text": text, "conf": conf, "bbox": bbox}


def obs():
    return {"frame_size": [W, H], "tracks": {
        "1": {"vehicle_class": "car", "last_seen_frame": 100, "votes": [
            vote(10, 0.9, [1, 100, 800, 700]),           # mejor lectura, pero cortada por el borde
            vote(5, 0.6, [200, 100, 900, 700])]},        # completa, menos confiable
        "2": {"vehicle_class": "car", "last_seen_frame": 200, "votes": [
            vote(50, 0.8, [100, 0, 900, 600]), vote(55, 0.7, [100, 1, 900, 600])]},   # todas cortadas
        "3": {"vehicle_class": "motorcycle", "last_seen_frame": 300, "votes": []},
    }}


def test_separates_complete_from_cut_readings():
    rows = {r["id_vehiculo"]: r for r in analyze_observations(obs())}
    assert rows[1]["lecturas"] == 2 and rows[1]["lecturas_completas"] == 1
    assert rows[1]["mejor_lectura"]["conf"] == 0.9 and rows[1]["mejor_lectura"]["toca_borde"]
    assert rows[1]["mejor_completa"]["conf"] == 0.6 and not rows[1]["mejor_completa"]["toca_borde"]
    assert rows[2]["lecturas_completas"] == 0 and rows[2]["mejor_completa"] is None
    assert rows[3]["lecturas"] == 0 and rows[3]["mejor_lectura"] is None      # sin lecturas: no falla


def test_report_text_summarizes():
    rows = analyze_observations(obs())
    txt = format_report(rows, [W, H], 5)
    assert "ID 1" in txt and "TOCA EL BORDE" in txt
    assert "al menos una lectura de caja completa: 1 de 3" in txt


def test_old_observations_without_frame_size_are_rejected():
    with pytest.raises(ValueError):
        analyze_observations({"tracks": {}})
