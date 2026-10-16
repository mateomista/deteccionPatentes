import json

from alpr.contracts import sanitize_video_stem as contract_sanitize
from alpr.tools.join_results import join_results, main, sanitize_video_stem


def det(id_, plate):
    return {"id_vehiculo": id_, "patente_validada": plate, "frame_idx": 10}


def cls(track, video, etiqueta="Ford_Fiesta", estado="ok"):
    return {"track_id": track, "video": video, "etiqueta": etiqueta, "marca": "Ford", "modelo": "Fiesta",
            "confianza": 0.9, "estado": estado}


def test_join_by_video_and_track_id():
    joined, no_class, orphans = join_results(
        [det(1, "AB123CD"), det(2, "EF456GH")], [cls(1, "v1"), cls(2, "OTRO"), cls(9, "v1")], "v1")
    assert joined[0]["clasificacion"]["etiqueta"] == "Ford_Fiesta"
    assert joined[1]["clasificacion"] is None                      # el ID 2 es de OTRO video
    assert [d["id_vehiculo"] for d in no_class] == [2]
    assert [o["track_id"] for o in orphans] == [9]


def test_same_track_id_in_two_videos_is_not_mixed_up():
    cl = [cls(1, "a", "Toyota_Hilux"), cls(1, "b", "Fiat_Cronos")]
    ja, _, _ = join_results([det(1, "AA111AA")], cl, "a")
    jb, _, _ = join_results([det(1, "BB222BB")], cl, "b")
    assert ja[0]["clasificacion"]["etiqueta"] == "Toyota_Hilux"
    assert jb[0]["clasificacion"]["etiqueta"] == "Fiat_Cronos"


def test_errors_are_not_joined_and_last_valid_wins():
    cl = [cls(1, "v", estado="error: x"), cls(1, "v", "Viejo"), cls(1, "v", "Nuevo")]
    joined, _, _ = join_results([det(1, "AB123CD")], cl, "v")
    assert joined[0]["clasificacion"]["etiqueta"] == "Nuevo"
    joined, no_class, _ = join_results([det(1, "AB123CD")], [cls(1, "v", estado="error: x")], "v")
    assert joined[0]["clasificacion"] is None and len(no_class) == 1


def test_legacy_names_need_explicit_opt_in():
    legacy = [cls(1, None)]
    assert join_results([det(1, "AB123CD")], legacy, "v")[0][0]["clasificacion"] is None
    assert join_results([det(1, "AB123CD")], legacy, "v", accept_legacy=True)[0][0]["clasificacion"]


def test_sanitize_matches_the_contract():
    for s in ("Exportación de ACC - 2026", "a:b*c", "///", "ok_1.2-3"):
        assert sanitize_video_stem(s) == contract_sanitize(s)


def test_cli(tmp_path, capsys):
    d = tmp_path / "trafico01.json"
    d.write_text(json.dumps([det(1, "AB123CD")]))
    c = tmp_path / "c.jsonl"
    c.write_text(json.dumps(cls(1, "trafico01")) + "\n\n" + "basura\n")
    out = tmp_path / "u.json"
    assert main(["--detections", str(d), "--classifications", str(c), "--out", str(out)]) == 0
    assert "clasificadas=1" in capsys.readouterr().out
    assert json.loads(out.read_text())[0]["clasificacion"]["marca"] == "Ford"
