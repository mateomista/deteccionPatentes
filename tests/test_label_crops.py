"""Herramienta de etiquetas: del índice de recortes de patente a labels.csv para ocr_eval."""
import csv
import json

import numpy as np
import pytest

from alpr.plate_crops import PlateCropExporter
from alpr.tools import label_crops as lc


def img(w=60, h=20):
    return np.full((h, w, 3), 120, np.uint8)


@pytest.fixture()
def crops_dir(tmp_path):
    d = tmp_path / "crops"
    ex = PlateCropExporter(d, "clip")
    ex.add("ocr_ok", 305, 10, img(228, 81), "car", text="AG353GI", conf=0.82)
    ex.add("ocr_sin_lectura", 287, 10, img(203, 41), "car", attempts=[])
    ex.add("ocr_ok", 306, 11, img(224, 79), "truck", text="AG253GI", conf=0.37)
    ex.add("ocr_ok", 83, 3, img(103, 28), "car", text="KDJ628", conf=0.78)
    ex.add("focus_mask", 390, 13, img(395, 55), "car", detail="aspecto")
    ex.add("sin_patente_vehiculo", 33, 3, img(371, 306), "car")
    ex.finish()
    return d


def test_plantilla_tiene_un_vehiculo_por_fila_y_no_muestra_lo_que_leyo_el_ocr(crops_dir, tmp_path):
    out = tmp_path / "vehiculos.csv"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(out)])
    text = out.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    assert [r["id_vehiculo"] for r in rows] == ["3", "10", "11"]      # el #13 solo tiene focus mask
    assert rows[1]["recortes_ocr"] == "2" and rows[1]["cuadros"] == "287-305"
    assert rows[1]["imagen_ejemplo"] == "muestras/ID10_F000305.jpg"          # el recorte más grande
    assert all(r["plate"] == "" for r in rows)
    assert "lecturas_ocr" not in rows[0]
    assert "AG353GI" not in text and "KDJ628" not in text                       # nada de lo que leyó el OCR
    samples = sorted(p.name for p in (tmp_path / "muestras").iterdir())
    assert samples == ["ID10_F000305.jpg", "ID11_F000306.jpg", "ID3_F000083.jpg"]
    assert not any("KDJ" in n or "AG" in n for n in samples)


def test_plantilla_con_lecturas_si_se_pide(crops_dir, tmp_path):
    out = tmp_path / "v.csv"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(out), "--show-reads"])
    rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines()))
    assert rows[0]["lecturas_ocr"] == "KDJ628"


def test_plantilla_no_pisa_un_archivo_existente(crops_dir, tmp_path):
    out = tmp_path / "v.csv"
    out.write_text("trabajo del usuario", encoding="utf-8")
    with pytest.raises(SystemExit):
        lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(out)])
    assert out.read_text(encoding="utf-8") == "trabajo del usuario"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(out), "--overwrite"])


def fill(path, plates, exclude=None, delimiter=","):
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    for r in rows:
        r["plate"] = plates.get(int(r["id_vehiculo"]), "")
        r["excluir_frames"] = (exclude or {}).get(int(r["id_vehiculo"]), "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)


def read_labels(out_dir):
    return list(csv.DictReader((out_dir / "labels.csv").read_text(encoding="utf-8").splitlines()))


def test_build_hereda_la_patente_a_los_recortes_de_cada_vehiculo(crops_dir, tmp_path, capsys):
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    fill(v, {10: "ab 953 gi", 11: "AB953GI", 3: "KDJ628"})
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    rows = read_labels(out)
    assert [(r["image"], r["plate"], r["vehicle_class"]) for r in rows] == [
        ("clip__F000083_ID3.jpg", "KDJ628", "car"),
        ("clip__F000287_ID10.jpg", "AB953GI", "car"),
        ("clip__F000305_ID10.jpg", "AB953GI", "car"),
        ("clip__F000306_ID11.jpg", "AB953GI", "truck"),
    ]
    assert all((out / r["image"]).exists() for r in rows)
    assert not (out / "labels.csv").read_bytes().startswith(b"\xef\xbb\xbf")     # ocr_eval lo lee sin BOM
    assert "AG" not in " ".join(r["image"] for r in rows)       # el nombre no filtra la lectura del OCR


def test_build_sin_patente_no_etiqueta_y_excluir_frames_saca_recortes(crops_dir, tmp_path):
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    fill(v, {10: "AB953GI"}, exclude={10: "287"})
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    assert [r["image"] for r in read_labels(out)] == ["clip__F000305_ID10.jpg"]
    assert not (out / "clip__F000287_ID10.jpg").exists()


def test_build_acepta_csv_con_punto_y_coma_de_excel(crops_dir, tmp_path):
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    fill(v, {3: "KDJ628"}, delimiter=";")
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    assert [r["plate"] for r in read_labels(out)] == ["KDJ628"]


def test_build_avisa_si_la_patente_no_tiene_formato_argentino(crops_dir, tmp_path, capsys):
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    fill(v, {3: "KDJ62"})
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    assert "no tiene formato de patente argentina" in capsys.readouterr().err
    assert [r["plate"] for r in read_labels(out)] == ["KDJ62"]      # avisa pero no decide por el usuario


def test_build_sin_ninguna_patente_falla_y_no_escribe_nada(crops_dir, tmp_path):
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    with pytest.raises(SystemExit):
        lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    assert not out.exists()


def test_build_se_puede_repetir_y_suma_sin_duplicar(crops_dir, tmp_path):
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    fill(v, {3: "KDJ628"})
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    fill(v, {3: "KDJ628", 11: "AB953GI"})
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    assert [r["image"] for r in read_labels(out)] == ["clip__F000083_ID3.jpg", "clip__F000306_ID11.jpg"]


def test_sin_indice_falla_con_un_mensaje_claro(tmp_path):
    with pytest.raises(SystemExit) as e:
        lc.main(["template", "--crops-dir", str(tmp_path), "--out", str(tmp_path / "v.csv")])
    assert "export-plate-crops" in str(e.value)


def test_labels_son_compatibles_con_ocr_eval(crops_dir, tmp_path):
    from alpr.evaluate.ocr_eval import load_labels
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    fill(v, {3: "KDJ628"})
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    rows = load_labels(str(out / "labels.csv"))
    assert rows[0]["image"] == "clip__F000083_ID3.jpg" and rows[0]["plate"] == "KDJ628"
    assert json.loads(json.dumps(rows))      # serializable, sin sorpresas


def test_build_frena_si_faltan_archivos_y_no_escribe_nada(crops_dir, tmp_path, capsys):
    v, out = tmp_path / "v.csv", tmp_path / "calib"
    lc.main(["template", "--crops-dir", str(crops_dir), "--out", str(v)])
    fill(v, {10: "AB953GI"})
    (crops_dir / "ocr_ok" / "F000305_ID10_AG353GI.jpg").unlink()
    with pytest.raises(SystemExit) as e:
        lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out)])
    assert "Faltan 1 archivos" in str(e.value)
    assert not out.exists()
    lc.main(["build", "--crops-dir", str(crops_dir), "--vehicles", str(v), "--out-dir", str(out),
             "--allow-missing"])
    assert [r["image"] for r in read_labels(out)] == ["clip__F000287_ID10.jpg"]
