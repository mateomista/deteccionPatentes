import json

import pytest

from alpr.data.ccpd import build_coco, parse_ccpd_bbox
from alpr.data.splits import assert_disjoint, load_split, main as splits_main, split_groups

CCPD_NAME = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"


def test_split_is_deterministic_disjoint_and_nonempty():
    groups = [f"g{i}" for i in range(20)]
    a = split_groups(groups, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    b = split_groups(groups, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    assert a == b
    assert_disjoint(a)
    assert sum(len(v) for v in a.values()) == 20
    assert all(len(v) >= 1 for v in a.values())
    assert split_groups(groups, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=2) != a


def test_split_fails_when_not_enough_groups():
    with pytest.raises(ValueError):
        split_groups(["a", "b"], {"train": 0.6, "val": 0.2, "test": 0.2})


def test_assert_disjoint_detects_leak():
    with pytest.raises(ValueError):
        assert_disjoint({"calib": ["v1", "v2"], "test": ["v2"]})


def test_video_split_cli_with_ground_truth(tmp_path):
    vd = tmp_path / "videos"
    vd.mkdir()
    for n in ("a", "b", "c", "d", "unlabeled"):
        (vd / f"{n}.mp4").write_bytes(b"")
    gt = tmp_path / "gt.csv"
    gt.write_text("video,plate,first_frame,last_frame\n" + "".join(f"{n},AB123CD,0,1\n" for n in "abcd"))
    out = tmp_path / "splits.json"
    splits_main(["videos", "--videos-dir", str(vd), "--ground-truth", str(gt),
                 "--split", "calib=0.5", "--split", "test=0.5", "--out", str(out)])
    calib, test = load_split(out, "calib"), load_split(out, "test")
    assert set(calib) | set(test) == set("abcd") and not set(calib) & set(test)


def test_coco_split_by_group_keeps_sequences_together(tmp_path):
    images = [{"id": i, "file_name": f"seq{i // 4}/f{i}.jpg", "width": 10, "height": 10} for i in range(40)]
    anns = [{"id": i, "image_id": i, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}
            for i in range(40)]
    src = tmp_path / "all.json"
    src.write_text(json.dumps({"images": images, "annotations": anns,
                               "categories": [{"id": 0, "name": "plate"}]}))
    out = tmp_path / "out"
    splits_main(["coco", "--annotations", str(src), "--split", "train=0.7", "--split", "val=0.15",
                 "--split", "test=0.15", "--group-regex", r"^(seq\d+)/", "--out-dir", str(out)])
    seqs = {}
    for name in ("train", "val", "test"):
        d = json.loads((out / f"{name}.json").read_text())
        assert len(d["annotations"]) == len(d["images"])
        seqs[name] = {i["file_name"].split("/")[0] for i in d["images"]}
    assert not (seqs["train"] & seqs["val"]) and not (seqs["train"] & seqs["test"]) \
        and not (seqs["val"] & seqs["test"])


def test_ccpd_bbox_parse():
    assert parse_ccpd_bbox(CCPD_NAME) == (154, 383, 386, 473)
    assert parse_ccpd_bbox("not-a-ccpd-name.jpg") is None


def test_build_coco_from_ccpd_names(tmp_path):
    from PIL import Image
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (720, 1160)).save(tmp_path / "sub" / CCPD_NAME)
    Image.new("RGB", (10, 10)).save(tmp_path / "sub" / "garbage.jpg")
    coco, skipped = build_coco(str(tmp_path))
    assert skipped == 1 and len(coco["images"]) == 1
    assert coco["annotations"][0]["bbox"] == [154, 383, 232, 90]
    assert coco["images"][0]["file_name"] == f"sub/{CCPD_NAME}"


def test_coco_split_groups_windows_style_paths_together(tmp_path):
    """Un COCO generado en Windows trae barras invertidas: las secuencias deben seguir juntas."""
    images = [{"id": i, "file_name": f"seq{i // 4}\\f{i}.jpg", "width": 10, "height": 10} for i in range(40)]
    anns = [{"id": i, "image_id": i, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}
            for i in range(40)]
    src = tmp_path / "all.json"
    src.write_text(json.dumps({"images": images, "annotations": anns,
                               "categories": [{"id": 0, "name": "plate"}]}))
    out = tmp_path / "out"
    splits_main(["coco", "--annotations", str(src), "--split", "train=0.7", "--split", "val=0.15",
                 "--split", "test=0.15", "--group-regex", r"^(seq\d+)/", "--out-dir", str(out)])
    seqs = {}
    for name in ("train", "val", "test"):
        d = json.loads((out / name).with_suffix(".json").read_text())
        seqs[name] = {i["file_name"].split("\\")[0] for i in d["images"]}
    assert not (seqs["train"] & seqs["val"]) and not (seqs["train"] & seqs["test"]) \
        and not (seqs["val"] & seqs["test"])


def test_to_posix_converts_windows_separators():
    from alpr.data.ccpd import to_posix
    assert to_posix("sub\\x.jpg") == "sub/x.jpg"
    assert to_posix("sub/x.jpg") == "sub/x.jpg"
