import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from alpr.data.splits import main as splits_main

IMG = np.zeros((20, 30, 3), np.uint8)


def make_source(tmp_path, spec):
    """spec: {clase: [nombres de archivo]}"""
    for cls, names in spec.items():
        (tmp_path / "raw" / cls).mkdir(parents=True, exist_ok=True)
        for n in names:
            cv2.imwrite(str(tmp_path / "raw" / cls / n), IMG)
    return tmp_path / "raw"


def run(src, out, *extra):
    splits_main(["classes", "--source", str(src), "--out-dir", str(out), "--split", "train=0.6",
                 "--split", "val=0.2", "--split", "test=0.2", *extra])


def test_same_vehicle_never_crosses_partitions_and_every_class_is_everywhere(tmp_path):
    spec = {"Toyota_Etios": [f"vehiculo_{i}__v{i % 2}.jpg" for i in range(30)] +
                            [f"vehiculo_{i}__v{i % 2}_b.jpg" for i in range(30)],
            "Peugeot_206": [f"vehiculo_{100 + i}__v1.jpg" for i in range(20)]}
    # cada vehículo tiene 2 recortes en Toyota (a y b): comparten el grupo "vehiculo_<id>__v<k>"
    src = make_source(tmp_path, spec)
    out = tmp_path / "out"
    run(src, out, "--group-regex", r"^(vehiculo_\d+__v\d)(?:_b)?\.jpg$")
    where = defaultdict(set)
    for split in ("train", "val", "test"):
        for cls in ("Toyota_Etios", "Peugeot_206"):
            for f in (out / split / cls).glob("*.jpg"):
                where[(cls, f.name.replace("_b", "").split(".")[0])].add(split)
    assert where and all(len(v) == 1 for v in where.values())        # ningún vehículo en dos particiones
    for cls in ("Toyota_Etios", "Peugeot_206"):
        for split in ("train", "val", "test"):
            assert any((out / split / cls).glob("*.jpg")), (cls, split)
    counts = json.loads((out / "splits_clasificacion.json").read_text(encoding="utf-8"))["counts"]
    assert sum(counts["Toyota_Etios"].values()) == 60 and sum(counts["Peugeot_206"].values()) == 20


def test_without_group_regex_each_file_is_its_own_group(tmp_path):
    src = make_source(tmp_path, {"A": [f"{i}.jpg" for i in range(10)]})
    run(src, tmp_path / "out")
    total = sum(len(list((tmp_path / "out" / s / "A").glob("*.jpg"))) if (tmp_path / "out" / s / "A").exists() else 0
                for s in ("train", "val", "test"))
    assert total == 10


def test_class_with_too_few_groups_goes_whole_to_the_biggest_split_with_a_warning(tmp_path, capsys):
    src = make_source(tmp_path, {"Rara": ["a.jpg", "b.jpg"], "Comun": [f"{i}.jpg" for i in range(12)]})
    run(src, tmp_path / "out")
    assert "Rara" in capsys.readouterr().err
    assert len(list((tmp_path / "out" / "train" / "Rara").glob("*.jpg"))) == 2
    assert not (tmp_path / "out" / "test" / "Rara").exists()


def test_group_in_two_classes_is_reported_as_a_labeling_error(tmp_path, capsys):
    src = make_source(tmp_path, {"A": ["vehiculo_1__v.jpg"] + [f"vehiculo_{i}__v.jpg" for i in range(2, 9)],
                                 "B": ["vehiculo_1__v.jpg"] + [f"vehiculo_{i}__w.jpg" for i in range(20, 27)]})
    run(src, tmp_path / "out", "--group-regex", r"^(vehiculo_\d+__.+)\.jpg$")
    assert "vehiculo_1__v" in capsys.readouterr().err


def test_refuses_to_write_into_a_non_empty_output_dir(tmp_path):
    import pytest
    src = make_source(tmp_path, {"A": ["1.jpg", "2.jpg", "3.jpg"]})
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "algo.txt").write_text("x")
    with pytest.raises(SystemExit):
        run(src, tmp_path / "out")
    assert Path(tmp_path / "out" / "algo.txt").exists()
