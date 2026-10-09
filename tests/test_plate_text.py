from alpr.plate_text import is_valid_plate, matches_official_format, select_ocr_result, smart_correct_plate


def frag(text, conf, x0, y0, x1, y1):
    return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, conf)


def test_strict_formats():
    for ok in ("ABC123", "AB123CD", "A123BCD"):
        assert is_valid_plate(ok, strict=True)
    for bad in ("ABCDEFG", "123456", "AB12CD", "", None):
        assert not is_valid_plate(bad, strict=True)


def test_permissive_accepts_numeric_only_but_strict_does_not():
    assert is_valid_plate("123456", strict=False)
    assert not is_valid_plate("123456", strict=True)


def test_smart_correct_fixes_digit_in_letter_slot():
    assert smart_correct_plate("A8C123") == "ABC123"     # 8 -> B en zona de letras
    assert smart_correct_plate("AB1Z3CD") == "AB123CD"   # Z -> 2 en zona de números


def test_smart_correct_tie_uses_vehicle_class():
    # "AAAAAAA" da la misma penalización como LLNNNLL (auto) y como LNNNLLL (moto).
    assert smart_correct_plate("AAAAAAA", "car") == "AA444AA"
    assert smart_correct_plate("AAAAAAA", "motorcycle") == "A444AAA"
    assert smart_correct_plate("") == ""


def test_select_ocr_result_merges_two_line_plate_top_to_bottom():
    top = frag("A123", 0.9, 10, 0, 90, 30)
    bottom = frag("BCD", 0.8, 10, 40, 90, 70)
    _, text, conf = select_ocr_result([bottom, top])
    assert text == "A123BCD"
    assert 0.8 < conf < 0.9


def test_select_ocr_result_drops_small_banner_text():
    plate = frag("AB123CD", 0.9, 10, 20, 200, 80)
    banner = frag("MERCOSUR", 0.7, 10, 0, 200, 12)
    assert select_ocr_result([banner, plate])[1] == "AB123CD"


def test_select_ocr_result_empty_and_lowconf():
    assert select_ocr_result([]) is None
    assert select_ocr_result([frag("X", 0.01, 0, 0, 10, 10)]) is None


def test_official_format_helper():
    assert matches_official_format("AB123CD") and not matches_official_format("AB12")
