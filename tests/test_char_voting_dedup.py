"""Voto por caracteres y deduplicación aproximada.

Los casos salen de un clip real: el mismo auto leído como KDV628, EDI628 y KDJ628 (tres textos
distintos, ninguno con dos votos), y el mismo vehículo contado dos veces como AG353GI y AG253GI."""
import numpy as np
import pytest

from alpr.config import Thresholds
from alpr.finalize import OUTCOME_DUPLICATE, OUTCOME_OK, find_recent_duplicate, prepare_record
from alpr.plate_text import fuse_plate_reads, plate_distance
from alpr.replay import replay_video
from alpr.tracking import TrackState, VehicleTrack

BBOX = np.array([0, 0, 100, 100])


def track(reads, char_voting=False, tid=1):
    t = VehicleTrack(tid, "car", char_voting=char_voting)
    t.state = TrackState.OCR_ACTIVE
    for i, (text, conf) in enumerate(reads):
        t.add_vote(text, conf, 10 + i, None, None, BBOX)
    return t


REAL_READS = [("KDV628", 0.22), ("EDI628", 0.54), ("KDJ628", 0.78)]


# ---------------------------------------------------------------- distancia y fusión
def test_plate_distance():
    assert plate_distance("AG253GI", "AG253GI") == 0
    assert plate_distance("AG253GI", "AG353GI") == 1
    assert plate_distance("KDJ628", "KDV628") == 1
    assert plate_distance("KDJ628", "KDJ62") == 1          # un carácter perdido
    assert plate_distance("KDJ628", "AB123CD") > 3


def test_fusion_de_las_lecturas_reales():
    assert fuse_plate_reads({t: c for t, c in REAL_READS}) == "KDJ628"


def test_fusion_puede_armar_un_texto_que_ninguna_lectura_tiene():
    reads = {"KDJ698": 0.5, "KDV628": 0.5, "EDJ628": 0.5}
    assert fuse_plate_reads(reads) == "KDJ628" and "KDJ628" not in reads


def test_fusion_usa_el_largo_con_mas_confianza_y_es_determinista():
    assert fuse_plate_reads({"ABC123": 0.9, "AB123CD": 0.3}) == "ABC123"
    assert fuse_plate_reads({"ABC123": 0.5, "ABD123": 0.5}) == "ABC123"     # empate: alfabético
    assert fuse_plate_reads({}) is None


# ---------------------------------------------------------------- consenso por caracteres
def test_sin_voto_por_caracteres_las_tres_lecturas_no_hacen_consenso():
    t = track(REAL_READS)
    assert t.get_consensus_plate() == ("KDJ628", pytest.approx(0.78), 1)
    assert t.check_finalization(Thresholds()) is False


def test_con_voto_por_caracteres_el_texto_gana_con_las_tres_lecturas_pero_no_adelanta_la_decision():
    t = track(REAL_READS, char_voting=True)
    plate, conf, count = t.get_consensus_plate()
    assert plate == "KDJ628" and count == 3 and conf == pytest.approx((0.22 + 0.54 + 0.78) / 3)
    # El consenso anticipado sigue exigiendo lecturas idénticas.
    assert t.check_finalization(Thresholds()) is False
    assert t.flush_eligible(Thresholds()) is True


def test_el_voto_por_caracteres_corrige_cuando_la_lectura_mas_confiable_esta_mal():
    reads = [("KDV628", 0.60), ("EDJ628", 0.55), ("KDJ698", 0.50)]    # ninguna es KDJ628
    assert track(reads).get_consensus_plate()[0] == "KDV628"          # texto entero: gana la más confiable
    plate, _, count = track(reads, char_voting=True).get_consensus_plate()
    assert plate == "KDJ628" and count == 3


def test_con_dos_lecturas_distintas_el_voto_por_caracteres_no_hace_consenso():
    reads = [("KDV628", 0.30), ("EDI628", 0.25)]
    t = track(reads, char_voting=True)
    assert t.get_consensus_plate() == track(reads).get_consensus_plate() == ("KDV628", pytest.approx(0.30), 1)
    assert t.flush_eligible(Thresholds()) is False      # una lectura con confianza < flush_single_vote_conf


def test_el_texto_armado_trae_evidencia_propia_y_consistente():
    t = track([("KDJ698", 0.6), ("KDV628", 0.5), ("EDJ628", 0.4)], char_voting=True)
    plate, _, count = t.get_consensus_plate()
    assert plate == "KDJ628" and count == 3
    ev = t.evidence.get(plate)
    assert ev is not None and ev.text == plate and ev.frame >= 0 and ev.bbox is not None
    # La evidencia sale de la lectura más parecida y más confiable (KDV628, distancia 1, conf 0.5
    # contra KDJ698 con distancia 1 y conf 0.6 -> gana la más confiable).
    assert ev.best_conf == pytest.approx(0.6)


def test_el_voto_por_caracteres_no_arma_un_texto_peor_que_una_lectura_real():
    # Las tres lecturas son oficiales; el texto fusionado (A1123CD) no lo es: se usa el voto
    # por texto entero.
    reads = [("AB123CD", 0.4), ("A123BCD", 0.4), ("AC123BD", 0.4)]
    assert fuse_plate_reads({t: c for t, c in reads}) == "A1123CD"
    t = track(reads, char_voting=True)
    plate, _, _ = t.get_consensus_plate()
    assert plate in dict(reads)


def test_con_una_sola_lectura_o_apagado_no_cambia_nada():
    one = track([("KDJ628", 0.8)], char_voting=True)
    assert one.get_consensus_plate() == ("KDJ628", pytest.approx(0.8), 1)
    same = [("AB123CD", 0.9), ("AB123CD", 0.8), ("XY987ZW", 0.2)]
    assert track(same, char_voting=True).get_consensus_plate()[0] == \
        track(same, char_voting=False).get_consensus_plate()[0] == "AB123CD"


def test_con_voto_por_caracteres_lecturas_de_otro_largo_no_cuentan():
    t = track([("KDJ628", 0.7), ("KDJ628", 0.6), ("AB123CD", 0.3)], char_voting=True)
    plate, _, count = t.get_consensus_plate()
    assert plate == "KDJ628" and count == 2


# ---------------------------------------------------------------- replay de punta a punta
def obs(tracks, fps=25.0, total=1000):
    return {"fps": fps, "strict_plate": True, "total_frames": total, "tracks": {
        str(tid): {"vehicle_class": "car", "last_seen_frame": 100,
                   "votes": [{"frame": 10 + i, "text": t, "conf": c, "bbox": [0, 0, 100, 100]}
                             for i, (t, c) in enumerate(reads)]}
        for tid, reads in tracks.items()}}


def test_replay_con_voto_por_caracteres_registra_la_patente_con_las_tres_lecturas():
    base = replay_video(obs({1: REAL_READS}), Thresholds())
    fused = replay_video(obs({1: REAL_READS}), Thresholds(), char_voting=True)
    assert [r["patente_validada"] for r in base] == ["KDJ628"] and base[0]["cantidad_votos"] == 1
    assert [r["patente_validada"] for r in fused] == ["KDJ628"]
    assert fused[0]["cantidad_votos"] == 3 and fused[0]["finalization"] != "consensus"


def test_replay_registra_una_patente_armada_sin_que_la_valide_otra_lectura():
    reads = [("KDJ698", 0.5), ("KDV628", 0.5), ("EDJ628", 0.5)]
    recs = replay_video(obs({1: reads}), Thresholds(), char_voting=True)
    assert [r["patente_validada"] for r in recs] == ["KDJ628"]


# ---------------------------------------------------------------- deduplicación aproximada
def done(plate, tid, frame, fps=25.0):
    t = track([(plate, 0.8), (plate, 0.8)], tid=tid)
    for ev in t.evidence.values():
        ev.frame = frame
    t.state = TrackState.FINALIZED
    return t


def register(plates, **kw):
    recent, outcomes = {}, []
    for i, (plate, frame) in enumerate(plates):
        _, outcome, detail = prepare_record(done(plate, i + 1, frame), "consensus", 25.0, True, recent,
                                            30.0, require_crop=False, **kw)
        outcomes.append((outcome, detail))
    return outcomes, recent


def test_dedup_exacta_sigue_igual_por_defecto():
    out, _ = register([("AG253GI", 100), ("AG253GI", 150)])
    assert [o for o, _ in out] == [OUTCOME_OK, OUTCOME_DUPLICATE]


def test_dedup_aproximada_une_el_mismo_vehiculo_leido_con_un_caracter_distinto():
    plates = [("AG253GI", 100), ("AG353GI", 150)]
    assert [o for o, _ in register(plates)[0]] == [OUTCOME_OK, OUTCOME_OK]               # apagada
    out, recent = register(plates, dedup_max_distance=1)
    assert [o for o, _ in out] == [OUTCOME_OK, OUTCOME_DUPLICATE]
    assert "AG253GI" in out[1][1] and "distancia 1" in out[1][1]
    assert set(recent) == {"AG253GI"}


def test_dedup_aproximada_respeta_la_distancia_la_ventana_y_el_largo():
    far = [("AG253GI", 100), ("AB123CD", 150)]
    assert [o for o, _ in register(far, dedup_max_distance=1)[0]] == [OUTCOME_OK, OUTCOME_OK]
    two = [("AG253GI", 100), ("AG353GJ", 150)]
    assert [o for o, _ in register(two, dedup_max_distance=1)[0]] == [OUTCOME_OK, OUTCOME_OK]
    assert [o for o, _ in register(two, dedup_max_distance=2)[0]] == [OUTCOME_OK, OUTCOME_DUPLICATE]
    late = [("AG253GI", 100), ("AG353GI", 100 + 25 * 31)]          # 31 s después, fuera de la ventana
    assert [o for o, _ in register(late, dedup_max_distance=1)[0]] == [OUTCOME_OK, OUTCOME_OK]
    other_len = [("KDJ628", 100), ("AB123CD", 150)]
    assert [o for o, _ in register(other_len, dedup_max_distance=2)[0]] == [OUTCOME_OK, OUTCOME_OK]


def test_find_recent_duplicate_prefiere_la_exacta_y_despues_la_mas_parecida():
    recent = {"AG253GI": 4.0, "AG353GJ": 5.0}
    assert find_recent_duplicate(recent, "AG253GI", 6.0, 30.0, 1) == ("AG253GI", 4.0)
    assert find_recent_duplicate(recent, "AG353GL", 6.0, 30.0, 1) == ("AG353GJ", 5.0)
    assert find_recent_duplicate({"AG253GI": 4.0, "AG353GI": 5.0}, "AG353GJ", 6.0, 30.0, 1) == ("AG353GI", 5.0)
    assert find_recent_duplicate(recent, "ZZ999ZZ", 6.0, 30.0, 2) is None


def test_replay_dedup_aproximada_cuenta_un_vehiculo_una_vez():
    both = obs({10: [("AG353GI", 0.82), ("AG353GI", 0.80)], 11: [("AG253GI", 0.76), ("AG253GI", 0.70)]})
    assert len(replay_video(both, Thresholds())) == 2
    assert len(replay_video(both, Thresholds(), dedup_max_distance=1)) == 1


# ---------------------------------------------------------------- pipeline real (modelos simulados)
class ScriptedReader:
    """EasyOCR simulado: devuelve las lecturas de `reads` una por llamada y después nada."""
    def __init__(self, reads):
        self.reads = list(reads)

    def readtext(self, image, **kw):
        if not self.reads:
            return []
        text, conf = self.reads.pop(0)
        return [([[0, 0], [100, 0], [100, 30], [0, 30]], text, conf)]


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("supervision")
    from tests.test_pipeline_smoke import make_video
    path = tmp_path_factory.mktemp("vid") / "trafico01.mp4"
    make_video(path)
    return str(path)


def run_pipeline(tmp_path, video, reads, **opts):
    from tests.test_pipeline_smoke import build
    pipe, _ = build(tmp_path, reader=ScriptedReader(reads), **opts)
    return pipe.run_video(video).detections


WRONG_BEST_READ = [("KDV628", 0.60), ("EDJ628", 0.55), ("KDJ698", 0.50)]


def test_pipeline_voto_por_caracteres_apagado_por_defecto(tmp_path, video):
    recs = run_pipeline(tmp_path, video, WRONG_BEST_READ)
    assert [r["patente_validada"] for r in recs] == ["KDV628"]


def test_pipeline_con_voto_por_caracteres_corrige_la_patente(tmp_path, video):
    recs = run_pipeline(tmp_path, video, WRONG_BEST_READ, char_voting=True)
    assert [r["patente_validada"] for r in recs] == ["KDJ628"]
    assert recs[0]["cantidad_votos"] == 3 and recs[0]["format_valid"] is True
