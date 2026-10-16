"""Entrenamiento del clasificador de punta a punta, con un modelo base diminuto y un dataset sintético.

Cada clase es un color distinto, así que el modelo aprende en segundos y el criterio de aceptación
(--min-accuracy) se puede probar de verdad. No mide si acierta marcas reales (no hay datos reales acá)."""
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from alpr.classify.classifier import VehicleClassifier  # noqa: E402
from alpr.train.train_classifier import check_license, main as train_main, normalize_license  # noqa: E402

CLASSES = {"Toyota_Etios": (200, 40, 40), "Peugeot_206": (40, 200, 40), "Ford_Fiesta": (40, 40, 200)}


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    d = tmp_path_factory.mktemp("base") / "tiny_vit"
    cfg = transformers.ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=64)
    transformers.set_seed(0)
    transformers.ViTForImageClassification(cfg).save_pretrained(d)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    return d


def write_dataset(root, per_split=None):
    per_split = per_split or {"train": 12, "val": 4, "test": 4}
    rng = np.random.default_rng(0)
    n = 0
    for split, count in per_split.items():
        for cls, color in CLASSES.items():
            (root / split / cls).mkdir(parents=True, exist_ok=True)
            for _ in range(count):
                n += 1
                img = np.clip(np.array(color, np.float32) + rng.normal(0, 25, (40, 40, 3)), 0, 255)
                # cada imagen lleva una marca única en una esquina: distintas entre sí, aunque de la misma clase
                img[: 1 + n % 6, : 1 + (n * 7) % 9] = rng.integers(0, 255, 3)
                cv2.imwrite(str(root / split / cls / f"img_{n}.png"), img.astype(np.uint8))
    return root


def args(data, base, out, *extra, base_license="apache-2.0", data_license="CC BY 4.0"):
    return ["--data-dir", str(data), "--base-model", str(base), "--base-model-license", base_license,
            "--dataset-license", data_license, "--output-dir", str(out), "--epochs", "8", "--batch-size", "8",
            "--lr", "5e-3", "--workers", "0", "--no-fp16", "--device", "cpu", *extra]


# -------------------------------------------------------------------------- licencias
def test_license_normalization_and_allowlist():
    assert normalize_license("CC BY 4.0") == "cc-by-4.0" and normalize_license("Apache_2.0") == "apache-2.0"
    assert check_license("x", "Apache-2.0") == "apache-2.0" and check_license("x", "MIT") == "mit"
    for bad in ("AGPL-3.0", "GPL-3.0", "cc-by-nc-4.0", "CC BY-SA 4.0", "research only", ""):
        with pytest.raises(ValueError):
            check_license("x", bad)


def test_non_permissive_license_aborts_before_training(tmp_path, tiny_base, capsys):
    data = write_dataset(tmp_path / "data")
    assert train_main(args(data, tiny_base, tmp_path / "out", base_license="cc-by-nc-4.0")) == 2
    assert "no permitida" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()                      # no entrenó ni escribió nada
    assert train_main(args(data, tiny_base, tmp_path / "out2", data_license="AGPL-3.0")) == 2


# -------------------------------------------------------------------- fuga de datos
def test_leakage_between_train_and_test_aborts(tmp_path, tiny_base, capsys):
    data = write_dataset(tmp_path / "data")
    src = next((data / "train" / "Toyota_Etios").glob("*.png"))
    (data / "test" / "Toyota_Etios" / "filtrada.png").write_bytes(src.read_bytes())
    assert train_main(args(data, tiny_base, tmp_path / "out")) == 2
    err = capsys.readouterr().err
    assert "FUGA DE DATOS" in err and "filtrada.png" in err
    assert not (tmp_path / "out" / "modelo").exists()


# ------------------------------------------------------------ entrenamiento y promoción
@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_base):
    root = tmp_path_factory.mktemp("entrenamiento")
    data = write_dataset(root / "data")
    out, promoted = root / "out", root / "promovido"
    rc = train_main(args(data, tiny_base, out, "--min-accuracy", "0.9", "--promote-to", str(promoted),
                         "--dataset-source", "dataset sintético de prueba"))
    return {"rc": rc, "out": out, "promoted": promoted, "data": data, "base": tiny_base, "root": root}


def test_training_learns_and_writes_metrics(trained):
    assert trained["rc"] == 0
    m = json.loads((trained["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert m["test"]["accuracy"] >= 0.9 and m["test"]["n"] == 12
    assert m["test"]["classes"] == sorted(CLASSES)
    assert all(v["recall"] is not None for v in m["test"]["per_class"].values())
    assert m["inference"]["median_ms"] > 0 and m["inference"]["device"] == "cpu"
    assert (trained["out"] / "confusion_matrix_test.csv").exists()


def test_provenance_records_licenses_and_criteria(trained):
    cfg = json.loads((trained["out"] / "train_config.json").read_text(encoding="utf-8"))
    assert cfg["licencia_modelo_base"] == "apache-2.0" and cfg["licencia_dataset"] == "cc-by-4.0"
    assert cfg["dataset_origen"] == "dataset sintético de prueba"
    assert cfg["clases"] == sorted(CLASSES) and cfg["imagenes"] == {"train": 36, "val": 12, "test": 12}
    assert cfg["criterio"]["promovido"] is True and cfg["criterio"]["minimo_exigido"] == 0.9
    assert cfg["versiones"]["torch"] and cfg["semilla"] == 42


def test_promoted_model_is_what_the_classifier_loads(trained, tmp_path):
    promoted = trained["promoted"]
    assert (promoted / "config.json").exists() and (promoted / "train_config.json").exists()
    clf = VehicleClassifier(promoted, "cpu")
    img = trained["data"] / "test" / "Peugeot_206"
    label, brand, model, conf = clf.classify(next(img.glob("*.png")))
    assert (label, brand, model) == ("Peugeot_206", "Peugeot", "206") and conf > 0.5


def test_weights_are_not_promoted_when_the_criterion_is_not_met(trained, tmp_path, capsys):
    out, promoted = tmp_path / "out", tmp_path / "promovido"
    rc = train_main(args(trained["data"], trained["base"], out, "--min-accuracy", "1.01",
                         "--promote-to", str(promoted)))
    assert rc == 3
    assert "NO PROMOVIDO" in capsys.readouterr().err
    assert not promoted.exists()                                  # nada llegó a weights/
    assert (out / "modelo" / "config.json").exists()              # pero el candidato quedó guardado
    cfg = json.loads((out / "train_config.json").read_text(encoding="utf-8"))
    assert cfg["criterio"]["promovido"] is False


def test_promote_to_requires_a_criterion(tmp_path, tiny_base):
    data = write_dataset(tmp_path / "data")
    with pytest.raises(SystemExit):
        train_main(args(data, tiny_base, tmp_path / "out", "--promote-to", str(tmp_path / "p")))


def test_no_test_split_means_no_promotion(tmp_path, tiny_base, capsys):
    data = write_dataset(tmp_path / "data", {"train": 12, "val": 4})
    rc = train_main(args(data, tiny_base, tmp_path / "out", "--min-accuracy", "0.5",
                         "--promote-to", str(tmp_path / "p")))
    assert rc == 3 and "no hay carpeta test" in capsys.readouterr().err
    assert not (tmp_path / "p").exists()
