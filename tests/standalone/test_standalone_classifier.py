"""Pruebas del clasificador STANDALONE (standalone/pipeclasificadorVehiculos.py).

Son las mismas que tests/classify/test_classify.py, ejecutadas contra el script suelto: así
las correcciones quedan verificadas en las dos versiones (modular y de un solo archivo).
"""
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from standalone_loader import load_standalone_classifier

C = load_standalone_classifier()
files_mod = processor_mod = C
main = C.main
extract_track_id, is_candidate, wait_until_file_is_stable = (
    C.extract_track_id, C.is_candidate, C.wait_until_file_is_stable)
parse_brand_model = C.parse_brand_model
FileProcessor, ClassificationResult, JsonlWriter = C.FileProcessor, C.ClassificationResult, C.JsonlWriter


# ------------------------------------------------------------------ utilidades
def make_image(path, seed=0):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
    return path


class FakeClassifier:
    """Clasificador 'real' simulado (mock=False) para no depender de ultralytics."""
    mock = False

    def classify(self, image_path):
        return ("Toyota_Corolla", "Toyota", "Corolla", 0.91)


@pytest.fixture
def fast_stability(monkeypatch):
    """La espera de estabilidad real puede tardar 10 s por archivo: en pruebas, medio segundo."""
    real = files_mod.wait_until_file_is_stable
    monkeypatch.setattr(
        processor_mod, "wait_until_file_is_stable",
        lambda path, stop_event, **kw: real(path, stop_event, timeout=0.6, interval=0.03,
                                            required_stable_checks=2))


def build(tmp_path, classifier=None, stop_event=None, **kw):
    inp = tmp_path / "in"
    inp.mkdir(exist_ok=True)
    writer = JsonlWriter(tmp_path / "out" / "res.jsonl")
    proc = FileProcessor(inp, classifier or FakeClassifier(), writer,
                         stop_event or threading.Event(), max_retries=kw.pop("max_retries", 2),
                         retry_delay=0.0, **kw)
    return inp, writer, proc


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------- etiquetas [HEREDADO]
@pytest.mark.parametrize("label,expected", [
    ("Toyota::Corolla", ("Toyota", "Corolla")),
    ("Mercedes-Benz|C200", ("Mercedes-Benz", "C200")),
    ("Ford/F_150", ("Ford", "F_150")),
    ("Ford_F_150", ("Ford", "F_150")),
    ("Volkswagen_Gol_Trend", ("Volkswagen", "Gol_Trend")),
    ("Mercedes_Benz_C200", ("Mercedes-Benz", "C200")),
    ("Alfa_Romeo_Giulia", ("Alfa Romeo", "Giulia")),
    ("MarcaX_ModeloY", ("MarcaX", "ModeloY")),
    ("", (None, None)),
    ("Desconocida_con_muchos_guiones", (None, None)),
])
def test_label_parser_inherited_behaviour(label, expected):
    assert parse_brand_model(label) == expected


def test_label_parser_non_string_is_controlled():
    assert parse_brand_model(None) == (None, None)


# ------------------------------------------------------- etiquetas [CORRECCIÓN]
@pytest.mark.parametrize("label,expected", [
    ("toyota", ("Toyota", None)),             # etiqueta de solo marca (clasificador por marca)
    ("Mercedes-Benz", ("Mercedes-Benz", None)),
    ("Citroën_C3", ("Citroen", "C3")),        # mismo resultado que citroen_c3
    ("citroen_c3", ("Citroen", "c3")),        # el modelo se conserva como viene escrito
    ("Citroën", ("Citroen", None)),
])
def test_label_parser_bare_brand_and_accents(label, expected):
    assert parse_brand_model(label) == expected


# ---------------------------------------------------- archivos candidatos [HEREDADO]
def test_candidate_rules(tmp_path):
    inp = tmp_path
    ok = make_image(inp / "vehiculo_17.jpg")
    assert is_candidate(ok, inp)
    assert extract_track_id(ok) == 17
    for bad in ("vehiculo_x.jpg", "otro_1.jpg", "vehiculo_1.txt", "~vehiculo_2.jpg", ".vehiculo_3.jpg"):
        (inp / bad).write_bytes(b"x")
        assert not is_candidate(inp / bad, inp), bad
    (inp / "vehiculo_5.jpg.tmp").write_bytes(b"x")
    assert not is_candidate(inp / "vehiculo_5.jpg.tmp", inp)
    (inp / "sub").mkdir()
    make_image(inp / "sub" / "vehiculo_9.jpg")
    assert not is_candidate(inp / "sub" / "vehiculo_9.jpg", inp)
    with pytest.raises(ValueError):
        extract_track_id(inp / "foto.jpg")


def test_stability_check_accepts_a_valid_image_in_a_non_ascii_directory(tmp_path):
    d = tmp_path / "Exportación ñandú"
    d.mkdir()
    img = make_image(d / "vehiculo_1.jpg")
    assert wait_until_file_is_stable(img, threading.Event(), timeout=2.0, interval=0.03,
                                     required_stable_checks=2)


def test_stability_check_rejects_corrupt_image(tmp_path):
    bad = tmp_path / "vehiculo_1.jpg"
    bad.write_bytes(b"esto no es una imagen")
    assert not wait_until_file_is_stable(bad, threading.Event(), timeout=0.4, interval=0.03,
                                         required_stable_checks=2)


# ------------------------------------------------- flujo completo (CLI) [HEREDADO]
def test_cli_mock_single_run_end_to_end(tmp_path, capsys):
    inp = tmp_path / "crops"
    inp.mkdir()
    for i in (1, 2, 3):
        make_image(inp / f"vehiculo_{i}.jpg", seed=i)
    (inp / "ignorado.jpg").write_bytes(b"x")
    res = tmp_path / "res.jsonl"
    main(["--input-dir", str(inp), "--results-file", str(res), "--mock", "--single-run", "--device", "cpu"])

    out = rows(res)
    assert sorted(r["track_id"] for r in out) == [1, 2, 3]
    assert all(r["estado"] == "mock" and r["etiqueta"] == "MOCK_VEHICULO" for r in out)
    assert sorted(p.name for p in (inp / "procesados").iterdir()) == [f"vehiculo_{i}.jpg" for i in (1, 2, 3)]
    assert (inp / "ignorado.jpg").exists()                       # no candidato: no se toca
    assert not any(inp.glob("vehiculo_*.jpg"))
    assert all(len(r["fingerprint_sha256"]) == 64 for r in out)


def test_idempotency_same_content_is_not_classified_twice(tmp_path):
    inp = tmp_path / "crops"
    inp.mkdir()
    res = tmp_path / "res.jsonl"
    argv = ["--input-dir", str(inp), "--results-file", str(res), "--mock", "--single-run", "--device", "cpu"]
    make_image(inp / "vehiculo_1.jpg", seed=1)
    main(argv)
    make_image(inp / "vehiculo_1.jpg", seed=1)    # mismo contenido, mismo nombre
    main(argv)
    assert len(rows(res)) == 1
    assert len(list((inp / "procesados").iterdir())) == 2        # el repetido también se archiva, sin pisar


def test_unreadable_file_goes_to_failed_and_is_recorded(tmp_path, fast_stability):
    inp, writer, proc = build(tmp_path, max_retries=1)
    (inp / "vehiculo_7.jpg").write_bytes(b"corrupta")
    proc.process_file((inp / "vehiculo_7.jpg").resolve())
    r = rows(writer.path)
    assert len(r) == 1 and r[0]["estado"].startswith("error:") and r[0]["track_id"] == 7
    assert (inp / "fallidos" / "vehiculo_7.jpg").exists()
    # un error no marca el hash como procesado: se podrá reintentar
    assert JsonlWriter(writer.path).load_processed_hashes() == set()


# ------------------------------------------------------- persistencia [CORRECCIÓN]
def test_mock_results_do_not_block_a_later_real_run(tmp_path, fast_stability):
    """Un recorte clasificado en modo --mock no debe contar como 'ya procesado' para el modelo real."""
    inp, writer, proc = build(tmp_path)
    img = make_image(inp / "vehiculo_1.jpg", seed=3)
    file_sha256 = C.file_sha256
    writer.append(ClassificationResult(1, str(img), str(img), "MOCK_VEHICULO", None, None, 0.0, "mock",
                                       file_sha256(img), "2026-01-01T00:00:00+00:00"))
    real_proc = FileProcessor(inp, FakeClassifier(), JsonlWriter(writer.path), threading.Event(),
                              max_retries=1, retry_delay=0.0)
    real_proc.process_file(img.resolve())
    estados = [r["estado"] for r in rows(writer.path)]
    assert estados == ["mock", "ok"]


def test_mock_run_still_skips_files_already_done_by_mock(tmp_path, fast_stability):
    inp, writer, proc = build(tmp_path)
    proc.classifier = type("M", (), {"mock": True, "classify": lambda s, p: ("MOCK_VEHICULO", None, None, 0.0)})()
    img = make_image(inp / "vehiculo_1.jpg", seed=3)
    proc.process_file(img.resolve())
    make_image(inp / "vehiculo_1.jpg", seed=3)
    proc2 = FileProcessor(inp, proc.classifier, JsonlWriter(writer.path), threading.Event(),
                          max_retries=1, retry_delay=0.0)
    proc2.process_file((inp / "vehiculo_1.jpg").resolve())
    assert len(rows(writer.path)) == 1


def test_jsonl_partial_last_line_does_not_corrupt_the_next_record(tmp_path):
    path = tmp_path / "res.jsonl"
    good = ClassificationResult(1, "a", "a", "x", None, None, 0.5, "ok", "f" * 64, "t")
    path.write_text(good.to_json() + "\n" + '{"track_id": 2, "imagen": "trunc', encoding="utf-8")
    writer = JsonlWriter(path)
    writer.append(ClassificationResult(3, "c", "c", "y", None, None, 0.6, "ok", "e" * 64, "t"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["track_id"] == 3             # el nuevo registro quedó en su propia línea
    assert json.loads(lines[0])["track_id"] == 1
    assert writer.load_processed_hashes() == {"f" * 64, "e" * 64}


def test_append_failure_after_move_is_retried_not_lost(tmp_path, fast_stability):
    inp, writer, proc = build(tmp_path, max_retries=2)
    calls = {"n": 0}
    real_append = writer.append

    def flaky(result):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError("disco ocupado")
        real_append(result)

    writer.append = flaky
    img = make_image(inp / "vehiculo_4.jpg", seed=4)
    proc.process_file(img.resolve())
    r = rows(writer.path)
    assert len(r) == 1 and r[0]["track_id"] == 4 and r[0]["estado"] == "ok"
    assert (inp / "procesados" / "vehiculo_4.jpg").exists()


def test_append_that_never_works_logs_the_record_so_it_can_be_recovered(tmp_path, fast_stability, caplog):
    inp, writer, proc = build(tmp_path, max_retries=1)
    writer.append = lambda result: (_ for _ in ()).throw(OSError("sin espacio"))
    img = make_image(inp / "vehiculo_5.jpg", seed=5)
    with caplog.at_level("ERROR", logger="clasificador_offline"):
        proc.process_file(img.resolve())
    assert any('"track_id": 5' in m for m in caplog.messages)    # el registro completo queda en el log


# ---------------------------------------------------------- apagado [CORRECCIÓN]
def test_shutdown_leaves_pending_files_in_place_instead_of_failing_them(tmp_path):
    """Ctrl+C no debe mandar a 'fallidos' los recortes que simplemente no llegaron a procesarse."""
    stop = threading.Event()
    inp, writer, proc = build(tmp_path, stop_event=stop)
    img = make_image(inp / "vehiculo_8.jpg", seed=8)
    stop.set()
    proc.process_file(img.resolve())
    assert img.exists()                                       # sigue en la carpeta de entrada
    assert not (inp / "fallidos").exists() or not any((inp / "fallidos").iterdir())
    assert not writer.path.exists() or rows(writer.path) == []


def test_files_not_enqueued_because_queue_is_full_are_picked_up_by_the_next_scan(tmp_path):
    """[HEREDADO] Con la cola llena, un archivo no encolado se recupera en el siguiente escaneo.
    (En modo watchdog no hay escaneo periódico: ver 'limitaciones conocidas' en el README.)"""
    inp, writer, proc = build(tmp_path)
    for i in (1, 2):
        make_image(inp / f"vehiculo_{i}.jpg", seed=i)
    import queue
    proc.queue = queue.Queue(maxsize=1)
    proc.scan_pending()                                       # entra 1; el otro no cabe
    assert proc.queue.qsize() == 1
    proc.queue.get_nowait(); proc.pending.clear()
    proc.scan_pending()                                       # el segundo se recupera
    assert proc.queue.qsize() == 1


# ------------------------------------------- contrato con el productor (video en el nombre)
def test_candidate_accepts_video_suffix_and_keeps_legacy_names(tmp_path):
    extract_video = C.extract_video
    new = make_image(tmp_path / "vehiculo_17__trafico01.jpg")
    old = make_image(tmp_path / "vehiculo_18.jpg")
    assert is_candidate(new, tmp_path) and is_candidate(old, tmp_path)
    assert (extract_track_id(new), extract_video(new)) == (17, "trafico01")
    assert (extract_track_id(old), extract_video(old)) == (18, None)
    (tmp_path / "vehiculo_19__trafico01.jpg.tmp").write_bytes(b"x")          # escritura en curso
    assert not is_candidate(tmp_path / "vehiculo_19__trafico01.jpg.tmp", tmp_path)


def test_result_carries_the_video_from_the_filename(tmp_path, fast_stability):
    inp, writer, proc = build(tmp_path)
    proc.process_file(make_image(inp / "vehiculo_5__trafico01.jpg", seed=1).resolve())
    proc.process_file(make_image(inp / "vehiculo_6.jpg", seed=2).resolve())
    r = rows(writer.path)
    assert [(x["track_id"], x["video"]) for x in r] == [(5, "trafico01"), (6, None)]


def test_error_record_also_carries_the_video(tmp_path, fast_stability):
    inp, writer, proc = build(tmp_path, max_retries=1)
    (inp / "vehiculo_7__v.jpg").write_bytes(b"corrupta")
    proc.process_file((inp / "vehiculo_7__v.jpg").resolve())
    r = rows(writer.path)[0]
    assert r["video"] == "v" and r["estado"].startswith("error:")
