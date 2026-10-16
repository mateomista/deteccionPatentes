import json

import pytest

from alpr.tools.compare_runs import compare_detections, load_detections, main, summarize


def det(plate, frame, id_=1, conf=0.9, votes=2, fin="consensus", cls="car"):
    return {"patente_validada": plate, "frame_idx": frame, "id_vehiculo": id_,
            "confianza_ocr_promedio": conf, "cantidad_votos": votes,
            "finalization": fin, "tipo_vehiculo": cls}


def test_identical_runs_are_parity():
    a = [det("AB123CD", 100, 1), det("EF456GH", 300, 2)]
    r = compare_detections(a, [dict(d) for d in a])
    s = summarize(r)
    assert s["matched"] == 2 and s["parity"] and s["max_frame_diff"] == 0


def test_small_frame_shift_is_parity_within_tolerance_but_not_when_strict():
    a, b = [det("AB123CD", 100)], [det("AB123CD", 103)]
    r = compare_detections(a, b)
    assert summarize(r, frame_tolerance=5)["parity"]
    s = summarize(r, frame_tolerance=0)
    assert not s["parity"] and s["frame_diff_over_tolerance"] == 1
    assert s["fields_that_differ_in_matches"]["frame_idx"] == 1


def test_plate_only_in_one_run():
    r = compare_detections([det("AB123CD", 100, 1), det("ZZ999ZZ", 900, 7)], [det("AB123CD", 100, 1)])
    s = summarize(r)
    assert s["only_a"] == 1 and s["only_b"] == 0 and not s["parity"]
    assert r["only_a"][0]["patente_validada"] == "ZZ999ZZ"


def test_different_reading_of_same_vehicle_is_not_a_match():
    # misma id de vehículo y patente distinta
    r = compare_detections([det("AB580BP", 200, 4)], [det("AB580BB", 205, 4)])
    s = summarize(r)
    assert s["matched"] == 0 and s["different_reading"] == 1 and not s["parity"]
    # sin id en común: se detecta igual por distancia de edición + cercanía de frames
    r = compare_detections([det("AB580BP", 200, 4)], [det("AB580BB", 210, 9)])
    assert summarize(r)["different_reading"] == 1
    # lejos en frames y sin id: son dos detecciones distintas
    r = compare_detections([det("AB580BP", 200, 4)], [det("AB580BB", 5000, 9)])
    s = summarize(r)
    assert s["only_a"] == 1 and s["only_b"] == 1


def test_same_plate_twice_pairs_by_closest_frame():
    a = [det("AB123CD", 100, 1), det("AB123CD", 2000, 5)]
    b = [det("AB123CD", 2003, 5), det("AB123CD", 98, 1)]
    r = compare_detections(a, b)
    assert summarize(r)["matched"] == 2 and summarize(r)["max_frame_diff"] == 3


def test_plate_normalization_ignores_spaces_and_case():
    r = compare_detections([det("ab 123 cd", 10)], [det("AB123CD", 10)])
    assert summarize(r)["parity"]


def test_load_json_and_jsonl(tmp_path):
    rows = [det("AB123CD", 1), det("EF456GH", 2)]
    j = tmp_path / "a.json"
    j.write_text(json.dumps(rows))
    jl = tmp_path / "a.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_detections(str(j)) == load_detections(str(jl)) == rows
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"foo": 1}]))
    with pytest.raises(ValueError):
        load_detections(str(bad))
    empty = tmp_path / "e.json"
    empty.write_text("")
    assert load_detections(str(empty)) == []


def test_cli_exit_codes_and_report(tmp_path, capsys):
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps([det("AB123CD", 100)]))
    b.write_text(json.dumps([det("AB123CD", 101)]))
    c.write_text(json.dumps([det("XX000XX", 100, 9)]))
    out = tmp_path / "r.json"
    assert main(["--a", str(a), "--b", str(b), "--label-a", "V5", "--label-b", "modular",
                 "--out", str(out)]) == 0
    assert "PARIDAD" in capsys.readouterr().out
    assert json.loads(out.read_text())["parity"] is True
    assert main(["--a", str(a), "--b", str(c)]) == 1
    txt = capsys.readouterr().out
    assert "CON DIFERENCIAS" in txt and "AB123CD" in txt and "XX000XX" in txt
    with pytest.raises(SystemExit):
        main(["--a", str(a), str(b), "--b", str(c)])    # distinta cantidad de archivos


def test_multiple_pairs_total(tmp_path, capsys):
    f = []
    for n in range(2):
        p = tmp_path / f"{n}.json"
        p.write_text(json.dumps([det("AB123CD", 10 * n)]))
        f.append(str(p))
    assert main(["--a", *f, "--b", *f]) == 0
    assert "PARIDAD en todos los videos" in capsys.readouterr().out
