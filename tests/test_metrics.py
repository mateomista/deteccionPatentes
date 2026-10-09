import pytest

from alpr.evaluate.metrics import (GTEntry, evaluate_predictions, levenshtein, load_ground_truth,
                                   match_video, ocr_metrics)


def gt(plate, a, b):
    return GTEntry("v", plate, a, b)


def pred(plate, frame, fin="consensus"):
    return {"patente_validada": plate, "frame_idx": frame, "finalization": fin}


def test_levenshtein():
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("ABC", "ABD") == 1
    assert levenshtein("", "AB") == 2


def test_match_categories():
    gts = [gt("AB123CD", 0, 100), gt("EF456GH", 200, 300)]
    preds = [pred("AB123CD", 50),      # tp
             pred("AB123CD", 60),      # duplicada
             pred("EF456GX", 250),     # errónea (1 carácter)
             pred("ZZ999ZZ", 500)]     # FP puro
    r = match_video(gts, preds)
    assert (r["tp"], r["duplicate"], r["wrong"], r["spurious"], r["fn"]) == (1, 1, 1, 1, 1)
    assert r["cer_edit"] == 1 and r["cer_chars"] == 14


def test_tolerance_frames():
    gts = [gt("AB123CD", 100, 200)]
    assert match_video(gts, [pred("AB123CD", 210)])["tp"] == 0
    assert match_video(gts, [pred("AB123CD", 210)], tolerance_frames=15)["tp"] == 1


def test_same_plate_two_passes_each_get_a_tp():
    gts = [gt("AB123CD", 0, 100), gt("AB123CD", 1000, 1100)]
    r = match_video(gts, [pred("AB123CD", 50), pred("AB123CD", 1050)])
    assert r["tp"] == 2 and r["fn"] == 0


def test_evaluate_predictions_precision_recall_and_filter():
    gts = {"v": [gt("AB123CD", 0, 100), gt("EF456GH", 200, 300)]}
    preds = {"v": [pred("AB123CD", 50), pred("EF456GH", 250, fin="end_of_video"), pred("QQ111QQ", 900)]}
    allm = evaluate_predictions(gts, preds)
    assert allm["tp"] == 2 and allm["spurious"] == 1
    assert allm["precision"] == pytest.approx(2 / 3) and allm["recall"] == 1.0
    cons = evaluate_predictions(gts, preds, finalizations={"consensus"})
    assert cons["tp"] == 1 and cons["recall"] == 0.5


def test_empty_cases_do_not_divide_by_zero():
    m = evaluate_predictions({"v": [gt("AB123CD", 0, 10)]}, {})
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["fn"] == 1


def test_ocr_metrics():
    m = ocr_metrics([("AB123CD", "AB123CD"), ("AB123CD", "AB123CX"), ("AB123CD", None)])
    assert m["n"] == 3 and m["n_read"] == 2 and m["n_exact"] == 1
    assert m["exact_match"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(0.5)
    assert m["cer_on_read"] == pytest.approx(1 / 14)


def test_load_ground_truth_validates(tmp_path):
    good = tmp_path / "gt.csv"
    good.write_text("video,plate,first_frame,last_frame\nv1,ab 123 cd,0,10\n", encoding="utf-8")
    assert load_ground_truth(good)["v1"][0].plate == "AB123CD"
    bad = tmp_path / "bad.csv"
    bad.write_text("video,plate,first_frame,last_frame\nv1,AB123CD,10,5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(bad)
    miss = tmp_path / "miss.csv"
    miss.write_text("video,plate\nv1,AB123CD\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(miss)
