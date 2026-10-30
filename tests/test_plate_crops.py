"""Exportación de recortes de patente con su resultado (alpr/plate_crops.py)."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("supervision")

from alpr.plate_crops import PlateCropExporter, render_html  # noqa: E402
from tests.test_pipeline_smoke import BadReader, build, make_video  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "trafico01.mp4"
    make_video(path)
    return str(path)


def _img(w=60, h=20):
    return np.full((h, w, 3), 128, np.uint8)


def test_exporter_guarda_imagen_indice_y_html(tmp_path):
    ex = PlateCropExporter(tmp_path / "out", "clip")
    rel = ex.add("ocr_ok", 12, 5, _img(), "car", text="AB123CD", conf=0.91,
                 attempts=[{"variant": "base", "texto_leido": "A8123CD", "texto_corregido": "AB123CD",
                            "resultado": "valida"}])
    ex.add("focus_mask", 13, 5, _img(), "car", detail="aspecto (7.18 fuera de [1.4, 5.5])")
    ex.add("sin_patente_vehiculo", 14, 6, _img(200, 100), "truck")
    counts = ex.finish()
    assert rel == "ocr_ok/F000012_ID5_AB123CD.jpg"
    assert (tmp_path / "out" / rel).exists()
    assert counts == {"ocr_ok": 1, "ocr_sin_lectura": 0, "focus_mask": 1, "sin_patente_vehiculo": 1}
    index = json.loads((tmp_path / "out" / "indice.json").read_text(encoding="utf-8"))
    assert len(index["recortes"]) == 3 and index["recortes"][0]["texto"] == "AB123CD"
    page = (tmp_path / "out" / "indice.html").read_text(encoding="utf-8")
    assert "Vehículo #5" in page and "Vehículo #6" in page
    assert "A8123CD → AB123CD" in page and "ocr_ok/F000012_ID5_AB123CD.jpg" in page


def test_exporter_ignora_imagenes_vacias_y_tipos_desconocidos(tmp_path):
    ex = PlateCropExporter(tmp_path, "clip")
    assert ex.add("ocr_ok", 1, 1, np.zeros((0, 0, 3), np.uint8), "car") is None
    assert ex.add("ocr_ok", 1, 1, None, "car") is None
    with pytest.raises(ValueError):
        ex.add("otro", 1, 1, _img(), "car")
    assert ex.items == []


def test_el_html_escapa_el_texto_leido():
    data = {"video": "<b>x</b>", "conteo": {"ocr_ok": 1, "ocr_sin_lectura": 0, "focus_mask": 0,
                                           "sin_patente_vehiculo": 0},
            "recortes": [{"archivo": "ocr_ok/a.jpg", "tipo": "ocr_ok", "frame": 1, "id_vehiculo": 1,
                          "tipo_vehiculo": "car", "texto": "<script>alert(1)</script>", "confianza": 0.5,
                          "detalle": None, "intentos": [], "ancho": 10, "alto": 5}]}
    page = render_html(data)
    assert "<script>" not in page and "&lt;script&gt;" in page and "<b>x</b>" not in page


def test_el_nombre_del_archivo_no_puede_salir_de_la_carpeta(tmp_path):
    ex = PlateCropExporter(tmp_path, "clip")
    rel = ex.add("ocr_ok", 1, 1, _img(), "car", text="../../x y\\z")
    assert rel.count("/") == 1 and "\\" not in rel and " " not in rel
    assert (tmp_path / rel).resolve().parent == (tmp_path / "ocr_ok").resolve()


def test_pipeline_exporta_lecturas_validas(tmp_path, video):
    out = tmp_path / "crops"
    pipe, _ = build(tmp_path, export_plate_dir=str(out))
    res = pipe.run_video(video)
    index = json.loads((out / "indice.json").read_text(encoding="utf-8"))
    kinds = {r["tipo"] for r in index["recortes"]}
    assert "ocr_ok" in kinds and (out / "indice.html").exists()
    assert all(r["texto"] == "AB123CD" for r in index["recortes"] if r["tipo"] == "ocr_ok")
    assert len(res.detections) >= 1


def test_pipeline_exporta_ocr_sin_lectura_con_los_intentos(tmp_path, video):
    out = tmp_path / "crops"
    pipe, _ = build(tmp_path, reader=BadReader(), export_plate_dir=str(out))
    pipe.run_video(video)
    index = json.loads((out / "indice.json").read_text(encoding="utf-8"))
    fails = [r for r in index["recortes"] if r["tipo"] == "ocr_sin_lectura"]
    assert fails and fails[0]["intentos"] and (out / fails[0]["archivo"]).exists()


def test_pipeline_guarda_el_vehiculo_cuando_rtdetr_no_encuentra_patente(tmp_path, video):
    out = tmp_path / "crops"
    pipe, _ = build(tmp_path, export_plate_dir=str(out))
    pipe._plate_forward = lambda crops: [None for _ in crops]
    pipe.run_video(video)
    index = json.loads((out / "indice.json").read_text(encoding="utf-8"))
    assert {r["tipo"] for r in index["recortes"]} == {"sin_patente_vehiculo"}


def test_sin_la_opcion_no_se_escribe_nada_y_las_detecciones_no_cambian(tmp_path, video):
    out = tmp_path / "crops"
    plain, _ = build(tmp_path)
    base = plain.run_video(video).detections
    with_export, _ = build(tmp_path, export_plate_dir=str(out))
    assert with_export.run_video(video).detections == base
    sin, _ = build(tmp_path)
    sin.run_video(video)
    assert sin.plate_exporter is None


def test_en_modo_grabacion_no_se_exporta(tmp_path, video):
    out = tmp_path / "crops"
    pipe, _ = build(tmp_path, record=True, export_plate_dir=str(out))
    pipe.run_video(video)
    assert not out.exists()
