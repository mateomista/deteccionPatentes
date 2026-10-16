"""El clasificador real (Transformers, Apache-2.0) en las DOS versiones: paquete y script suelto.

Usa un modelo diminuto con pesos aleatorios creado en un directorio temporal (sin descargas): no
verifica que acierte marcas, sino que cargue desde un directorio, clasifique una imagen, devuelva
etiqueta/marca/modelo/confianza coherentes y funcione de punta a punta dentro del procesador.
"""
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from standalone_loader import load_standalone_classifier  # noqa: E402

LABELS = ["Toyota_Etios", "Peugeot_206", "Mercedes-Benz", "Ford_F_150", "Desconocida_con_muchos_guiones"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("clasificador") / "vehicle_classifier"
    cfg = transformers.ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=64,
                                 id2label=dict(enumerate(LABELS)), label2id={n: i for i, n in enumerate(LABELS)})
    transformers.set_seed(0)
    model = transformers.ViTForImageClassification(cfg)
    model.save_pretrained(d)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def implementations():
    from alpr.classify import classifier as modular
    return {"paquete": modular, "standalone": load_standalone_classifier()}


@pytest.fixture(params=["paquete", "standalone"])
def impl(request, implementations):
    return implementations[request.param]


def make_image(path, seed=0):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (60, 90, 3), dtype=np.uint8))
    return path


def test_loads_from_a_directory_and_classifies(impl, tiny_model_dir, tmp_path):
    clf = impl.VehicleClassifier(tiny_model_dir, "cpu")
    label, brand, model, conf = clf.classify(make_image(tmp_path / "vehiculo_1.jpg"))
    assert label in LABELS
    assert 0.0 < conf <= 1.0
    # marca y modelo salen del MISMO parser que ya estaba probado
    expected = {"Toyota_Etios": ("Toyota", "Etios"), "Peugeot_206": ("Peugeot", "206"),
                "Mercedes-Benz": ("Mercedes-Benz", None), "Ford_F_150": ("Ford", "F_150"),
                "Desconocida_con_muchos_guiones": (None, None)}
    assert (brand, model) == expected[label]


def test_is_deterministic_and_probabilities_are_valid(impl, tiny_model_dir, tmp_path):
    clf = impl.VehicleClassifier(tiny_model_dir, "cpu")
    img = make_image(tmp_path / "vehiculo_2.jpg", seed=3)
    assert clf.classify(img) == clf.classify(img)


def test_different_images_can_give_different_confidences(impl, tiny_model_dir, tmp_path):
    clf = impl.VehicleClassifier(tiny_model_dir, "cpu")
    a = clf.classify(make_image(tmp_path / "a.jpg", seed=1))
    b = clf.classify(make_image(tmp_path / "b.jpg", seed=2))
    assert a[3] != b[3]                       # la entrada realmente influye en la salida


def test_missing_model_directory_explains_how_to_continue(impl, tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        impl.VehicleClassifier(tmp_path / "no_existe", "cpu")
    assert "--mock" in str(e.value)


def test_undecodable_image_is_an_error_not_a_silent_result(impl, tiny_model_dir, tmp_path):
    bad = tmp_path / "vehiculo_3.jpg"
    bad.write_bytes(b"no es una imagen")
    with pytest.raises(RuntimeError):
        impl.VehicleClassifier(tiny_model_dir, "cpu").classify(bad)


def test_image_in_a_non_ascii_directory(impl, tiny_model_dir, tmp_path):
    d = tmp_path / "Exportación ñandú"
    d.mkdir()
    label, *_ = impl.VehicleClassifier(tiny_model_dir, "cpu").classify(make_image(d / "vehiculo_4.jpg"))
    assert label in LABELS


def test_mock_mode_needs_no_model(impl, tmp_path):
    assert impl.VehicleClassifier(tmp_path / "no_existe", "cpu", mock=True).classify(tmp_path / "x.jpg") == \
           ("MOCK_VEHICULO", None, None, 0.0)


def test_real_classifier_inside_the_processor_end_to_end(tiny_model_dir, tmp_path):
    """El flujo completo con el clasificador REAL: recorte -> clasificación -> JSONL (estado 'ok')."""
    C = load_standalone_classifier()
    inp = tmp_path / "in"
    inp.mkdir()
    make_image(inp / "vehiculo_9__trafico01.jpg", seed=5)
    real = C.wait_until_file_is_stable
    C.wait_until_file_is_stable = lambda p, ev, **kw: real(p, ev, timeout=2.0, interval=0.03, required_stable_checks=2)
    try:
        writer = C.JsonlWriter(tmp_path / "res.jsonl")
        proc = C.FileProcessor(inp, C.VehicleClassifier(tiny_model_dir, "cpu"), writer, threading.Event(),
                               max_retries=1, retry_delay=0.0)
        proc.process_file((inp / "vehiculo_9__trafico01.jpg").resolve())
    finally:
        C.wait_until_file_is_stable = real
    row = json.loads(Path(tmp_path / "res.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["estado"] == "ok" and row["etiqueta"] in LABELS and row["video"] == "trafico01"
    assert 0.0 < row["confianza"] <= 1.0
