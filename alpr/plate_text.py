"""Validación, corrección y fusión de texto de patentes (sin dependencias de modelos)."""
import re

import numpy as np

from .config import (
    PLATE_REGEX_PATTERNS, DEBUG_PLATE_REGEX, STRICT_PLATE_VALIDATION,
    OCR_MIN_FRAGMENT_CONF, OCR_MIN_HEIGHT_RATIO, OCR_LINE_SPLIT_RATIO,
    OCR_MAX_MERGED_LEN,
)


def matches_official_format(text):
    """True si el texto cumple alguno de los formatos oficiales (regex estricto)."""
    if not text:
        return False
    return any(p.match(text) for p in PLATE_REGEX_PATTERNS)


def is_valid_plate(text, strict=None):
    """Valida una lectura contra los formatos de patente argentina.

    strict=True (producción, --strict-plate): exige LLLNNN, LLNNNLL o LNNNLLL,
    de modo que una mancha no pueda pasar por letra donde va un número. ES la
    validación de formato argentino.
    strict=False (SOLO debug): longitud 6-7 y alfanumérico en mayúsculas. NO
    equivale a formato argentino válido: "ABCDEFG" o "123456" pasan.
    strict=None usa STRICT_PLATE_VALIDATION. [FIX-v5 #7] El pipeline pasa el
    valor de --strict-plate.
    """
    if not text:
        return False
    if strict is None:
        strict = STRICT_PLATE_VALIDATION
    if strict:
        return matches_official_format(text)
    # [FIX-v3 #7] Antes: len(text) in (6,7) and text.isalnum() and text.isupper().
    # isupper() es False cuando no hay caracteres con caja, así que "123456" se
    # rechazaba por un motivo equivocado (y no por no ser una patente válida).
    return bool(DEBUG_PLATE_REGEX.match(text))


def smart_correct_plate(text, vehicle_class=None):
    """Aplica correcciones OCR solo donde el formato lo justifique.

    [FIX-v5 #8] Con 7 caracteres, LLNNNLL (auto) y LNNNLLL (moto) pueden dar la
    misma penalización. Antes el empate iba siempre a auto; ahora, si el
    vehículo es una moto, el empate se resuelve a favor de LNNNLLL.
    """
    if not text:
        return text

    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    length = len(text)

    # Mapa conservador: solo confusiones visualmente muy parecidas en chapas
    # argentinas, para no introducir correcciones erróneas.
    map_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '4': 'A', '6': 'G'}
    map_to_number = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'A': '4', 'G': '6'}

    def apply_format(raw, mask):
        corrected = []
        changes = 0
        for char, m in zip(raw, mask):
            if m == 'L':
                if char.isalpha():
                    corrected.append(char)
                elif char in map_to_letter:
                    corrected.append(map_to_letter[char])
                    changes += 1
                else:
                    corrected.append(char)
                    changes += 10  # penalización pesada si no se puede arreglar
            else:  # m == 'N'
                if char.isdigit():
                    corrected.append(char)
                elif char in map_to_number:
                    corrected.append(map_to_number[char])
                    changes += 1
                else:
                    corrected.append(char)
                    changes += 10
        return "".join(corrected), changes

    if length == 6:
        corrected, _ = apply_format(text, "LLLNNN")
        return corrected
    if length == 7:
        c1, penalties1 = apply_format(text, "LLNNNLL")
        c2, penalties2 = apply_format(text, "LNNNLLL")
        if penalties1 == penalties2:
            return c2 if vehicle_class == "motorcycle" else c1
        return c1 if penalties1 < penalties2 else c2

    return text


def _fragment_geometry(result):
    """Devuelve (cx, cy, alto) de un fragmento de EasyOCR."""
    xs = [p[0] for p in result[0]]
    ys = [p[1] for p in result[0]]
    return float(np.mean(xs)), float(np.mean(ys)), float(max(ys) - min(ys))


def select_ocr_result(ocr_results):
    """Fusiona los fragmentos de EasyOCR que componen UNA patente.

    Antes se devolvía solo el fragmento de mayor confianza, así que una chapa
    con separación visual ("AB 123 CD") devolvía medio dominio.

    [FIX-v3 #3] La v2 fusionaba únicamente dentro de la misma línea horizontal,
    lo que dejaba rotas las chapas de moto, que son de DOS líneas. Ahora:
      1) Se descartan fragmentos de confianza ínfima.
      2) Se descartan los fragmentos bajos respecto del más alto: así se va el
         texto chico de la banda azul (MERCOSUR / ARGENTINA) y los fantasmas,
         que son la fuente clásica de caracteres de más.
      3) Se agrupan en líneas por posición vertical, se ordenan las líneas de
         arriba hacia abajo y los fragmentos de izquierda a derecha.
      4) Si la concatenación completa supera OCR_MAX_MERGED_LEN, no es una
         patente: se cae a la línea que contiene el fragmento más confiable.

    La confianza devuelta es el promedio ponderado por longitud de texto.
    """
    if not ocr_results:
        return None

    frags = [r for r in ocr_results if r[2] >= OCR_MIN_FRAGMENT_CONF and r[1]]
    if not frags:
        return None
    if len(frags) == 1:
        return frags[0]

    geoms = {id(r): _fragment_geometry(r) for r in frags}
    max_h = max(geoms[id(r)][2] for r in frags)
    if max_h > 0:
        frags = [r for r in frags if geoms[id(r)][2] >= OCR_MIN_HEIGHT_RATIO * max_h]
    if not frags:
        return None

    # Agrupar en líneas por proximidad vertical.
    frags.sort(key=lambda r: geoms[id(r)][1])
    lines = []
    current = [frags[0]]
    for prev, cur in zip(frags, frags[1:]):
        _, prev_cy, prev_h = geoms[id(prev)]
        _, cur_cy, cur_h = geoms[id(cur)]
        ref_h = max(prev_h, cur_h, 1.0)
        if (cur_cy - prev_cy) > OCR_LINE_SPLIT_RATIO * ref_h:
            lines.append(current)
            current = [cur]
        else:
            current.append(cur)
    lines.append(current)

    for line in lines:
        line.sort(key=lambda r: min(p[0] for p in r[0]))

    def build(selected_lines):
        picked = [r for line in selected_lines for r in line]
        text = "".join(r[1] for r in picked)
        weights = [max(len(r[1]), 1) for r in picked]
        conf = float(np.average([r[2] for r in picked], weights=weights))
        anchor = max(picked, key=lambda r: r[2])
        return (anchor[0], text, conf)

    merged = build(lines)
    if len(merged[1]) <= OCR_MAX_MERGED_LEN:
        return merged

    # Demasiado texto para ser una patente: quedarse con la línea del fragmento
    # más confiable en vez de devolver una concatenación sin sentido.
    best_frag = max(frags, key=lambda r: r[2])
    for line in lines:
        # [FIX-v5 #9] Identidad, no igualdad de tuplas con listas de coordenadas.
        if any(r is best_frag for r in line):
            return build([line])
    return best_frag
