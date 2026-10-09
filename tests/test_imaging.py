import cv2
import numpy as np

from alpr.imaging import (has_plate_characteristics, prewarp_plate, preprocess_plate_for_ocr,
                          variant_clahe, variant_clahe_unsharp, variant_sauvola_strong)


def synthetic_plate(h=40, w=130, dark_bg=False):
    img = np.full((h, w, 3), 30 if dark_bg else 230, np.uint8)
    color = (235, 235, 235) if dark_bg else (20, 20, 20)
    cv2.putText(img, "AB123CD", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


def test_preprocess_returns_binary_upscaled_image():
    out = preprocess_plate_for_ocr(synthetic_plate())
    assert out is not None and out.ndim == 2
    assert set(np.unique(out)) <= {0, 255}
    assert out.shape[0] >= 48        # ampliado hasta OCR_MIN_HEIGHT


def test_polarity_dark_plate_ends_with_dark_text_on_light_background():
    out = preprocess_plate_for_ocr(synthetic_plate(dark_bg=True))
    assert out is not None
    assert (out == 255).mean() > 0.5   # fondo claro dominante tras normalizar


def test_variants_are_independent_and_do_not_mutate_input():
    crop = synthetic_plate()
    before = crop.copy()
    for fn in (variant_clahe, variant_clahe_unsharp, variant_sauvola_strong):
        out = fn(crop)
        assert out is not None and out.ndim == 2
    assert np.array_equal(crop, before)


def test_prewarp_expands_canvas_instead_of_cropping():
    crop = synthetic_plate(60, 200)
    M = cv2.getRotationMatrix2D((100, 30), 8, 1.0)
    tilted = cv2.warpAffine(crop, M, (200, 60), borderMode=cv2.BORDER_REPLICATE)
    out = prewarp_plate(tilted)
    assert out is not tilted                       # sí corrigió (8° está en el rango)
    assert out.shape[0] > tilted.shape[0] and out.shape[1] > tilted.shape[1]
    straight = prewarp_plate(crop)
    assert straight.shape[:2] == crop.shape[:2]    # sin inclinación: no toca nada


def test_prewarp_handles_empty_and_none():
    assert prewarp_plate(None) is None
    assert prewarp_plate(np.zeros((0, 0, 3), np.uint8)).size == 0


def test_focus_mask_geometry_by_vehicle_class():
    square = np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    assert has_plate_characteristics(square, "motorcycle")[0]        # casi cuadrada: ok en moto
    ok, reason = has_plate_characteristics(square, "car")
    assert not ok and "aspecto" in reason
    flat = np.full((30, 120, 3), 128, np.uint8)
    ok, reason = has_plate_characteristics(flat, "car")
    assert not ok and "desvio_gris" in reason
    assert not has_plate_characteristics(np.zeros((5, 5, 3), np.uint8), "car")[0]
