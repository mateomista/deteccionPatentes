"""Detección de FUGA de datos entre particiones de un dataset de clasificación.

Si la misma imagen (o una casi idéntica, p. ej. otro frame del mismo auto) cae en entrenamiento y
en prueba, la exactitud de prueba se infla y no mide nada. Esta comprobación busca, entre
particiones distintas:
  - copias exactas (mismo SHA-256);
  - imágenes casi idénticas (hash perceptual dHash de 64 bits con distancia de Hamming baja).

Estructura esperada: <raíz>/<partición>/<clase>/<imagen>. Para la búsqueda de casi-duplicados no se
comparan todos contra todos: se indexa por trozos del hash (principio del palomar), que encuentra
exactamente los mismos pares que la comparación completa pero en tiempo casi lineal.
"""
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(image_bgr, size=8):
    """Hash perceptual de diferencias: 64 bits que cambian poco ante recompresión o brillo."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a, b):
    return bin(a ^ b).count("1")


def _read(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_images(split_dirs):
    """{partición: carpeta} -> lista de dicts {path, split, sha, dhash}. Ignora archivos ilegibles."""
    rows = []
    for split, folder in split_dirs.items():
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix.lower() not in IMG_EXTS or not path.is_file():
                continue
            img = _read(path)
            if img is None or img.size == 0:
                continue
            rows.append({"path": str(path), "split": split, "sha": _sha256(path), "dhash": dhash(img)})
    return rows


def find_duplicates_between_splits(rows, hamming_threshold=2):
    """Pares de imágenes de PARTICIONES DISTINTAS que son copias exactas o casi idénticas."""
    found, seen = [], set()

    def add(a, b, kind, dist):
        key = tuple(sorted((a["path"], b["path"])))
        if a["split"] != b["split"] and key not in seen:
            seen.add(key)
            found.append({"a": a["path"], "split_a": a["split"], "b": b["path"], "split_b": b["split"],
                          "tipo": kind, "distancia": dist})

    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r["sha"]].append(r)
    for group in by_sha.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                add(group[i], group[j], "exacta", 0)

    # Casi-duplicados: con distancia <= t, al partir el hash en t+1 trozos al menos uno coincide.
    t = int(hamming_threshold)
    if t >= 0:
        bits, chunks = 64, t + 1
        edges = [round(i * bits / chunks) for i in range(chunks + 1)]
        buckets = defaultdict(list)
        for idx, r in enumerate(rows):
            for c in range(chunks):
                lo, hi = edges[c], edges[c + 1]
                buckets[(c, (r["dhash"] >> lo) & ((1 << (hi - lo)) - 1))].append(idx)
        checked = set()
        for members in buckets.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (members[i], members[j])
                    if pair in checked:
                        continue
                    checked.add(pair)
                    a, b = rows[pair[0]], rows[pair[1]]
                    d = hamming(a["dhash"], b["dhash"])
                    if d <= t and a["sha"] != b["sha"]:
                        add(a, b, "casi idéntica", d)
    return found


def find_cross_split_duplicates(split_dirs, hamming_threshold=2):
    return find_duplicates_between_splits(index_images(split_dirs), hamming_threshold)
