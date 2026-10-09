"""OCR adaptativo sobre UN crop de patente ([OCR-ADAPT]).

Se separó del pipeline para poder evaluarlo solo (alpr.evaluate.ocr_eval) con
un conjunto de recortes etiquetados, sin cargar los detectores ni correr video.
"""
import logging
from collections import defaultdict

from .config import (
    OCR_ALLOWLIST, OCR_TEXT_THRESHOLD, OCR_LOW_TEXT, OCR_MAX_ATTEMPTS,
    OCR_SUFFICIENT_CONF, OCR_VARIANT_BASE, OCR_VARIANT_ORDER,
)
from .imaging import OCR_VARIANT_BUILDERS, prewarp_plate, preprocess_plate_for_ocr
from .plate_text import (
    is_valid_plate, matches_official_format, select_ocr_result, smart_correct_plate,
)


def new_ocr_stats():
    """Contadores del OCR. Invariante:
    ocr_attempts_total == ocr_valid_readings + ocr_rejected_format
                          + ocr_rejected_confidence + ocr_no_text"""
    return {
        "ocr_crops_processed": 0,
        "ocr_crops_resolved": 0,
        "ocr_attempts_total": 0,
        "ocr_attempts_by_variant": defaultdict(int),
        "ocr_selected_by_variant": defaultdict(int),
        "ocr_valid_readings": 0,
        "ocr_rejected_format": 0,
        "ocr_rejected_confidence": 0,
        "ocr_no_text": 0,
    }


class PlateOCR:
    """Intento base + hasta 3 variantes de rescate independientes.

    candidate_min_conf: piso de confianza por candidato. Por defecto
    `thresholds.min_avg_ocr_conf`. El modo de grabación para calibración pasa
    0.0 para registrar TODAS las lecturas de formato válido y decidir el piso
    offline.
    """

    def __init__(self, reader, thresholds, strict_plate, dual_ocr=True,
                 max_attempts=OCR_MAX_ATTEMPTS, candidate_min_conf=None, stats=None):
        self.reader = reader
        self.strict = bool(strict_plate)
        self.candidate_min_conf = (thresholds.min_avg_ocr_conf
                                   if candidate_min_conf is None else float(candidate_min_conf))
        max_total = 1 + len(OCR_VARIANT_ORDER)
        self.max_attempts = min(max_total, max(1, int(max_attempts)))
        self.variants = OCR_VARIANT_ORDER[:self.max_attempts - 1] if dual_ocr else ()
        self.stats = stats if stats is not None else new_ocr_stats()
        for k, v in new_ocr_stats().items():
            self.stats.setdefault(k, v)

    def _ocr_pass(self, image):
        if image is None or image.size == 0:
            return None
        try:
            results = self.reader.readtext(
                image, detail=1, allowlist=OCR_ALLOWLIST,
                text_threshold=OCR_TEXT_THRESHOLD, low_text=OCR_LOW_TEXT)
        except Exception as e:
            logging.debug(f"[ocr] Pasada fallida: {e}")
            return None
        return select_ocr_result(results)

    def _attempt(self, variant, image, vehicle_class):
        """UN intento. Devuelve un dict candidato (formato Y confianza válidos) o None."""
        if image is None or image.size == 0:
            return None
        s = self.stats
        s["ocr_attempts_total"] += 1
        s["ocr_attempts_by_variant"][variant] += 1

        best = self._ocr_pass(image)
        if best is None:
            s["ocr_no_text"] += 1
            return None

        raw_text, conf = best[1], float(best[2])
        text = smart_correct_plate(raw_text, vehicle_class)

        if not is_valid_plate(text, self.strict):
            s["ocr_rejected_format"] += 1
            return None
        if conf < self.candidate_min_conf:
            s["ocr_rejected_confidence"] += 1
            return None

        s["ocr_valid_readings"] += 1
        return {"text": text, "conf": conf, "variant": variant,
                "processed": image, "official": matches_official_format(text)}

    def read(self, plate_crop, vehicle_class):
        """Devuelve (texto, confianza, imagen_procesada) con el ÚNICO candidato
        final, o None si ningún intento fue válido."""
        s = self.stats
        s["ocr_crops_processed"] += 1

        warped = prewarp_plate(plate_crop)   # base geométrica común
        candidates = []

        cand = self._attempt(OCR_VARIANT_BASE, preprocess_plate_for_ocr(warped), vehicle_class)
        if cand is not None:
            candidates.append(cand)

        for variant in self.variants:
            if any(c["conf"] >= OCR_SUFFICIENT_CONF for c in candidates):
                break
            cand = self._attempt(variant, OCR_VARIANT_BUILDERS[variant](warped), vehicle_class)
            if cand is not None:
                candidates.append(cand)

        if not candidates:
            return None

        winner = max(candidates, key=lambda c: (c["official"], c["conf"]))
        s["ocr_crops_resolved"] += 1
        s["ocr_selected_by_variant"][winner["variant"]] += 1
        return winner["text"], winner["conf"], winner["processed"]
