import pytest

from alpr.evaluate.classification_metrics import classification_report


def test_perfect_and_partial_accuracy():
    classes = ["A", "B", "C"]
    r = classification_report([0, 1, 2, 2], [0, 1, 2, 2], classes)
    assert r["accuracy"] == 1.0 and r["macro_recall"] == 1.0
    r = classification_report([0, 0, 1, 2], [0, 1, 1, 0], classes)
    assert r["accuracy"] == pytest.approx(0.5)
    assert r["per_class"]["A"]["recall"] == pytest.approx(0.5)
    assert r["per_class"]["A"]["precision"] == pytest.approx(0.5)      # predijo A 2 veces, 1 acierto
    assert r["per_class"]["C"]["recall"] == 0.0
    assert r["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_classes_without_examples_or_predictions_are_none_not_zero():
    r = classification_report([0, 0], [0, 0], ["A", "B"])
    assert r["per_class"]["B"]["recall"] is None and r["per_class"]["B"]["precision"] is None
    assert r["macro_recall"] == 1.0                    # la clase sin ejemplos no baja el promedio


def test_empty_and_mismatched_inputs():
    assert classification_report([], [], ["A"])["accuracy"] is None
    with pytest.raises(ValueError):
        classification_report([0], [0, 1], ["A", "B"])
