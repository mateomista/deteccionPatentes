import pytest

from alpr.evaluate.detection_metrics import average_precision, iou, precision_recall_at


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_perfect_detector_has_ap_1():
    gts = [[[0, 0, 10, 10]], [[5, 5, 20, 20]]]
    preds = [[([0, 0, 10, 10], 0.9)], [([5, 5, 20, 20], 0.8)]]
    assert average_precision(preds, gts) == pytest.approx(1.0)


def test_false_positive_ranked_first_lowers_ap():
    gts = [[[0, 0, 10, 10]]]
    preds = [[([50, 50, 60, 60], 0.95), ([0, 0, 10, 10], 0.5)]]
    assert average_precision(preds, gts) == pytest.approx(0.5)


def test_precision_recall_at_threshold():
    gts = [[[0, 0, 10, 10]], [[0, 0, 10, 10]]]
    preds = [[([0, 0, 10, 10], 0.9)], [([0, 0, 10, 10], 0.2)]]
    r = precision_recall_at(preds, gts, score_thr=0.5)
    assert r["precision"] == 1.0 and r["recall"] == 0.5 and r["fn"] == 1


def test_no_predictions():
    assert average_precision([[]], [[[0, 0, 1, 1]]]) == 0.0
