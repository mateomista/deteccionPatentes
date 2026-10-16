import random

import cv2
import numpy as np

from alpr.data.leakage import (dhash, find_cross_split_duplicates, find_duplicates_between_splits, hamming)


def photo(seed, size=(120, 90)):
    """Imagen con estructura (no ruido blanco), distinta por semilla."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    return cv2.resize(base, size, interpolation=cv2.INTER_CUBIC)


def save(path, img, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def dirs(tmp_path):
    return {n: tmp_path / n for n in ("train", "val", "test")}


def test_exact_copy_across_splits_is_found(tmp_path):
    save(tmp_path / "train/A/x.jpg", photo(1))
    (tmp_path / "test/A").mkdir(parents=True)
    (tmp_path / "test/A/copia.jpg").write_bytes((tmp_path / "train/A/x.jpg").read_bytes())
    save(tmp_path / "val/A/otra.jpg", photo(2))
    found = find_cross_split_duplicates(dirs(tmp_path))
    assert len(found) == 1 and found[0]["tipo"] == "exacta"
    assert {found[0]["split_a"], found[0]["split_b"]} == {"train", "test"}


def test_near_duplicate_survives_recompression_and_brightness(tmp_path):
    img = photo(3)
    save(tmp_path / "train/A/x.jpg", img, 95)
    save(tmp_path / "test/A/x_otro_frame.jpg", np.clip(img.astype(int) + 8, 0, 255).astype(np.uint8), 60)
    found = find_cross_split_duplicates(dirs(tmp_path), hamming_threshold=4)
    assert len(found) == 1 and found[0]["tipo"] == "casi idéntica" and found[0]["distancia"] <= 4


def test_different_images_and_same_split_duplicates_are_not_reported(tmp_path):
    save(tmp_path / "train/A/a.jpg", photo(10))
    save(tmp_path / "test/A/b.jpg", photo(11))
    save(tmp_path / "train/B/a_repetida.jpg", photo(10))              # repetida, pero DENTRO de train
    assert find_cross_split_duplicates(dirs(tmp_path)) == []


def test_unreadable_files_are_skipped(tmp_path):
    (tmp_path / "train/A").mkdir(parents=True)
    (tmp_path / "train/A/rota.jpg").write_bytes(b"no es una imagen")
    save(tmp_path / "test/A/ok.jpg", photo(1))
    assert find_cross_split_duplicates(dirs(tmp_path)) == []


def test_pigeonhole_index_finds_exactly_what_brute_force_finds():
    rng = random.Random(0)
    rows = []
    base = rng.getrandbits(64)
    for i in range(60):
        h = base
        for bit in rng.sample(range(64), rng.choice([0, 1, 2, 3, 5, 9, 20])):
            h ^= 1 << bit                                            # distancia variada respecto de base
        rows.append({"path": f"p{i}", "split": "train" if i % 2 else "test", "sha": f"s{i}", "dhash": h})
    for t in (0, 1, 2, 4):
        fast = {tuple(sorted((d["a"], d["b"]))) for d in find_duplicates_between_splits(rows, t)}
        brute = {tuple(sorted((a["path"], b["path"]))) for i, a in enumerate(rows) for b in rows[i + 1:]
                 if a["split"] != b["split"] and hamming(a["dhash"], b["dhash"]) <= t}
        assert fast == brute, f"umbral {t}"


def test_dhash_is_stable_for_identical_images():
    assert dhash(photo(5)) == dhash(photo(5)) and dhash(photo(5)) != dhash(photo(6))
