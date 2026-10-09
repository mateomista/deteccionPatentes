"""Operaciones de imagen del pipeline ALPR (prewarp, focus mask, binarización,
limpieza de contornos y variantes de rescate del OCR adaptativo).

Funciones puras sobre arrays de numpy/OpenCV: no cargan modelos, así que se
pueden testear y usar (p. ej. para generar figuras de la tesis) sin GPU.
"""
import logging

import cv2
import numpy as np

from .config import (
    OCR_MIN_HEIGHT, OCR_UPSCALE_MAX, POLARITY_MEDIAN_THRESHOLD,
    SAUVOLA_WINDOW, SAUVOLA_K, SAUVOLA_R,
    PREWARP_MIN_ANGLE, PREWARP_MAX_ANGLE,
    PLATE_GEOMETRY, MIN_GRAY_STD,
    MIN_CONTOUR_AREA_ABS, MIN_CONTOUR_AREA_RATIO, BORDER_MARGIN,
    MIN_CHAR_HEIGHT_RATIO, HOLE_MAX_AREA_RATIO, HOLE_MAX_HEIGHT_RATIO,
    BILATERAL_DIAMETER, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, UNSHARP_SIGMA, UNSHARP_AMOUNT,
    SAUVOLA_STRONG_K, SAUVOLA_STRONG_WINDOW_FACTOR, SAUVOLA_STRONG_CLAHE_CLIP,
)


def _ensure_gray(img):
    """Guarda de ndim: cvtColor(BGR2GRAY) explota si la imagen ya es 2D."""
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _ocr_scale_factor(height):
    """Factor de ampliación necesario para llegar a OCR_MIN_HEIGHT, con tope."""
    if height <= 0 or height >= OCR_MIN_HEIGHT:
        return 1.0
    return min(OCR_UPSCALE_MAX, float(OCR_MIN_HEIGHT) / float(height))


def _resize(img, scale):
    if img is None or img.size == 0 or scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img,
                      (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_CUBIC)


def _normalize_polarity(gray):
    """[FIX-v5 #5] Lleva la chapa a texto oscuro sobre fondo claro, sobre el GRIS.

    En una chapa el fondo ocupa la mayor parte de los píxeles, así que la
    MEDIANA del gris identifica la polaridad de forma robusta: mediana baja =>
    chapa de fondo oscuro con letras claras => se invierte.

    Esto tiene que ocurrir ANTES de binarizar. Sauvola manda las regiones
    uniformes a blanco (con std=0, T = media*(1-k) < píxel), por lo que una
    chapa negra sale mayormente blanca de la binarización y la media de la
    binaria ya no dice nada sobre su polaridad original.
    """
    if gray is None or gray.size == 0:
        return gray
    if float(np.median(gray)) < POLARITY_MEDIAN_THRESHOLD:
        return cv2.bitwise_not(gray)
    return gray


# ==============================================================================
# PRE/POST PROCESAMIENTO (mitigaciones de la tesis)
# ==============================================================================
def _sauvola_threshold(gray, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K, r=SAUVOLA_R):
    """Binarización local de Sauvola: T = mean * (1 + k * (std / R - 1)).

    A diferencia de un umbral global (Otsu), calcula un umbral por píxel con las
    estadísticas de su vecindad, así que resiste sombras proyectadas y
    degradados suaves de iluminación sobre la chapa.
    """
    gray = _ensure_gray(gray)
    window_size = int(window_size)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1  # la ventana de la media móvil debe tener centro

    gray_f = gray.astype(np.float32)
    ksize = (window_size, window_size)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=ksize, normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=ksize, normalize=True,
                            borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    threshold = mean * (1.0 + k * ((std / r) - 1.0))
    return np.where(gray_f > threshold, 255, 0).astype(np.uint8)


def prewarp_plate(plate_crop):
    """Endereza virtualmente la chapa antes del OCR (mitiga O vs Q).

    El ángulo sale del lado más largo del rectángulo mínimo (boxPoints +
    arctan2) en vez de rect[2]: la convención de ángulos de cv2.minAreaRect
    cambió en OpenCV 4.5 (hoy devuelve [0, 90)), así que el clásico
    `if angle < -45: angle = 90 + angle` puede reportar 88° para una chapa
    inclinada 2° y el filtro de rango la descarta sin que nadie se entere.

    [FIX-v3 #10] El lienzo se expande al bounding box de la rotación. Antes se
    rotaba dentro del mismo alto/ancho, de modo que con ángulos cercanos a los
    20° permitidos los caracteres de los extremos se salían del encuadre: el
    prewarp arreglaba la inclinación y de paso borraba la primera letra.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop
    try:
        gray = _ensure_gray(plate_crop)
        edges = cv2.Canny(gray, 50, 150)
        coords = cv2.findNonZero(edges)
        if coords is None or len(coords) < 10:
            return plate_crop

        box = cv2.boxPoints(cv2.minAreaRect(coords))
        d01 = float(np.linalg.norm(box[1] - box[0]))
        d12 = float(np.linalg.norm(box[2] - box[1]))
        edge = (box[1] - box[0]) if d01 >= d12 else (box[2] - box[1])
        angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0

        if abs(angle) < PREWARP_MIN_ANGLE or abs(angle) > PREWARP_MAX_ANGLE:
            return plate_crop

        h, w = plate_crop.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Lienzo expandido para no cortar las esquinas.
        cos_a = abs(matrix[0, 0])
        sin_a = abs(matrix[0, 1])
        new_w = int(round(h * sin_a + w * cos_a))
        new_h = int(round(h * cos_a + w * sin_a))
        matrix[0, 2] += (new_w / 2.0) - cx
        matrix[1, 2] += (new_h / 2.0) - cy

        return cv2.warpAffine(plate_crop, matrix, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        logging.debug(f"[prewarp] Error recuperable, se usa el crop original: {e}")
        return plate_crop


def has_plate_characteristics(crop, vehicle_class="default"):
    """Focus mask: filtro barato ANTES de gastar cómputo de OCR.

    Descarta regiones demasiado chicas (proxy de distancia extrema), con aspecto
    fuera del rango esperado, o uniformes (fondo del auto, sin texto). Acá nace
    el ahorro de cómputo: cuesta microsegundos y evita invocar EasyOCR, la etapa
    más cara por crop.

    [FIX-v3 #2] La geometría esperada depende de la clase de vehículo: las
    chapas de moto son de dos líneas y casi cuadradas, y con el rango único de
    la v2 (aspecto >= 1.4) quedaban descartadas antes del OCR.

    [DIAG] Devuelve (paso, motivo) en vez de solo bool, para poder loguear
    EXACTAMENTE por que se descarta un crop que RT-DETR si detecto. "motivo"
    es None cuando paso=True.
    """
    if crop is None or crop.size == 0 or crop.ndim < 2:
        return False, "crop_vacio"

    min_w, min_h, aspect_min, aspect_max = PLATE_GEOMETRY.get(
        vehicle_class, PLATE_GEOMETRY["default"])

    h, w = crop.shape[:2]
    if w < min_w or h < min_h:
        return False, f"tamano ({w}x{h} < minimo {min_w}x{min_h})"

    aspect = w / max(h, 1)
    if aspect < aspect_min or aspect > aspect_max:
        return False, f"aspecto ({aspect:.2f} fuera de [{aspect_min}, {aspect_max}])"

    gray = _ensure_gray(crop)
    std = float(np.std(gray))
    if std < MIN_GRAY_STD:
        return False, f"desvio_gris ({std:.1f} < minimo {MIN_GRAY_STD})"  # region uniforme => no hay texto

    return True, None


def _clean_binary_contours(binary):
    """Limpieza de contornos sobre la imagen binaria.

    Mitiga los ruidos propios de las chapas argentinas:
      1) Componentes con área despreciable: puntos sueltos y restos del texto
         fantasma gris MERCOSUR/ARGENTINA.
      2) Componentes que tocan el borde: marco de la chapa (que el OCR leía como
         "1" o "I") y sellos circulares RNPA de las chapas previas a 2016.
         Excepción: si el componente es alto respecto del crop y no atraviesa
         todo el ancho/alto, se conserva — un dígito pegado al borde por un
         recorte ajustado es un caso frecuente, no un marco.
      3) [FIX-v3 #1] Huecos internos: se rellenan SOLO los diminutos.

    Sobre el punto 3, que es el defecto más serio que arrastraban las tres
    versiones anteriores: rellenar todo hueco interno que no tocara el borde
    tapa la contraforma de O, 0, D, Q, 8, B, A, P y R. El resultado es que
    EasyOCR recibía blobs sólidos donde antes había caracteres distinguibles, y
    el sistema perdía justamente la distinción O vs Q que el prewarp existe para
    proteger. Ahora un hueco se rellena solo si es chico en área relativa Y bajo
    en altura relativa, que es el perfil del ruido de sal y pimienta dentro del
    trazo, no el de la contraforma de un glifo.

    [FIX-v5 #10] La reconstrucción de la máscara usa una tabla de búsqueda
    (keep[labels]) en vez de `labels == i` por componente: mismo resultado.

    Convención de entrada/salida: texto oscuro sobre fondo claro.
    """
    if binary is None or binary.size == 0:
        return binary
    try:
        binary = _ensure_gray(binary)
        fg = cv2.bitwise_not(binary)  # texto = 255, fondo = 0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        h, w = fg.shape

        # [FIX-v3 #9] Umbral relativo al recorte, con piso absoluto: un valor
        # fijo cambia de significado según cuánto se haya ampliado el crop.
        min_area = max(MIN_CONTOUR_AREA_ABS, int(MIN_CONTOUR_AREA_RATIO * h * w))
        keep = np.zeros(num, dtype=np.uint8)

        for i in range(1, num):  # 0 es el fondo
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]

            if area < min_area:
                continue

            touches_border = (
                x <= BORDER_MARGIN or y <= BORDER_MARGIN or
                (x + ww) >= (w - BORDER_MARGIN) or (y + hh) >= (h - BORDER_MARGIN)
            )
            if touches_border:
                height_ratio = hh / h if h > 0 else 0.0
                spans_full_width = ww >= (w - 2 * BORDER_MARGIN)
                spans_full_height = hh >= (h - 2 * BORDER_MARGIN) and ww < (0.5 * w)
                looks_like_frame_or_seal = spans_full_width or spans_full_height
                looks_like_character = (height_ratio >= MIN_CHAR_HEIGHT_RATIO
                                        and not looks_like_frame_or_seal)
                if not looks_like_character:
                    continue

            keep[i] = 255

        cleaned_fg = keep[labels]

        # Relleno acotado de huecos: solo ruido dentro del trazo.
        inv = cv2.bitwise_not(cleaned_fg)
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        max_hole_area = max(MIN_CONTOUR_AREA_ABS, int(HOLE_MAX_AREA_RATIO * h * w))
        max_hole_height = HOLE_MAX_HEIGHT_RATIO * h
        fill = np.zeros(num2, dtype=np.uint8)

        for i in range(1, num2):
            x = stats2[i, cv2.CC_STAT_LEFT]
            y = stats2[i, cv2.CC_STAT_TOP]
            ww = stats2[i, cv2.CC_STAT_WIDTH]
            hh = stats2[i, cv2.CC_STAT_HEIGHT]
            area = stats2[i, cv2.CC_STAT_AREA]

            is_internal = (x > BORDER_MARGIN and y > BORDER_MARGIN and
                           (x + ww) < (w - BORDER_MARGIN) and (y + hh) < (h - BORDER_MARGIN))
            if not is_internal:
                continue
            if area > max_hole_area or hh > max_hole_height:
                continue  # es la contraforma de un glifo: NO tocar
            fill[i] = 255

        cleaned_fg = cv2.bitwise_or(cleaned_fg, fill[labels2])

        return cv2.bitwise_not(cleaned_fg)
    except Exception as e:
        logging.warning(f"[clean] Limpieza de contornos falló, se usa la binarizada cruda: {e}")
        return binary


def preprocess_plate_for_ocr(plate_crop):
    """Cadena de preprocesamiento que ve EasyOCR (NO la evidencia visual).

    [OCR-ADAPT] Es el intento 1 ("base") del OCR adaptativo; no cambió.

    Orden:
      1) [FIX-v3 #5] Escalado sobre el GRIS. La v2 ampliaba al final, sobre la
         imagen ya binarizada: INTER_CUBIC sobre una binaria devuelve valores
         intermedios (la imagen deja de ser binaria) y amplifica el escalonado
         de los bordes. Ampliar antes le da además a Sauvola más píxeles reales
         por trazo con los que trabajar.
      2) [FIX-v5 #5] Normalización de polaridad sobre el gris: las Mercosur son
         fondo blanco / letras negras y las viejas fondo negro / letras blancas.
         Se decide con la mediana del gris ANTES de binarizar (después de
         Sauvola ya no es confiable, ver _normalize_polarity).
      3) Filtro bilateral: aplana ruido sin destruir los bordes de los caracteres.
      4) Sauvola con la ventana escalada en la misma proporción, para que siga
         cubriendo la misma cantidad de trazo que a escala original.
      5) Limpieza de contornos.

    Devuelve binaria lista para EasyOCR, o None si el preprocesamiento falló.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    try:
        gray = _ensure_gray(plate_crop)

        scale = _ocr_scale_factor(gray.shape[0])
        gray = _resize(gray, scale)

        gray = _normalize_polarity(gray)

        gray = cv2.bilateralFilter(gray, BILATERAL_DIAMETER,
                                   BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

        window = int(round(SAUVOLA_WINDOW * scale))
        binary = _sauvola_threshold(gray, window, SAUVOLA_K, SAUVOLA_R)

        return _clean_binary_contours(binary)
    except Exception as e:
        logging.warning(f"[preproc] Preprocesamiento de patente falló: {e}")
        return None


# ------------------------------------------------------------------------------
# [OCR-ADAPT] Variantes de rescate. Cada una es una función PURA e
# INDEPENDIENTE: recibe la misma base geométrica (plate_crop natural, ya
# enderezado por el prewarp) y arma su propia cadena de filtros desde cero.
# Ninguna recibe la salida de otra, y ninguna modifica su entrada in situ.
# ------------------------------------------------------------------------------
def _gray_base_for_ocr(plate_crop):
    """Base común de las variantes: gris ampliado y con polaridad normalizada.

    Es el mismo tratamiento previo al filtrado que usa el intento base (escala
    + polaridad), de modo que las variantes difieren SOLO en el filtrado.
    Devuelve (gris, escala) o (None, 1.0).
    """
    gray = _ensure_gray(plate_crop)
    if gray is None or gray.size == 0:
        return None, 1.0
    scale = _ocr_scale_factor(gray.shape[0])
    gray = _resize(gray, scale)
    return _normalize_polarity(gray), scale


def variant_clahe(plate_crop):
    """Gris + CLAHE (normalización local de contraste)."""
    try:
        gray, _ = _gray_base_for_ocr(plate_crop)
        if gray is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        return clahe.apply(gray)
    except Exception as e:
        logging.debug(f"[ocr_variant:clahe] falló: {e}")
        return None


def variant_clahe_unsharp(plate_crop):
    """Gris + CLAHE + Unsharp Mask (realza bordes de trazo borroso)."""
    try:
        gray, _ = _gray_base_for_ocr(plate_crop)
        if gray is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        eq = clahe.apply(gray)
        blur = cv2.GaussianBlur(eq, (0, 0), UNSHARP_SIGMA)
        return cv2.addWeighted(eq, 1.0 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0)
    except Exception as e:
        logging.debug(f"[ocr_variant:clahe_unsharp] falló: {e}")
        return None


def variant_sauvola_strong(plate_crop):
    """Binarización Sauvola reforzada: CLAHE fuerte + suavizado + Sauvola con
    k y ventana mayores que el intento base + limpieza de contornos.

    Se diferencia del intento base en el pre-filtro (CLAHE + gaussiano en vez de
    bilateral) y en los parámetros de Sauvola, para cubrir chapas con
    iluminación despareja o trazo grueso donde el base falla.
    """
    try:
        gray, scale = _gray_base_for_ocr(plate_crop)
        if gray is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=SAUVOLA_STRONG_CLAHE_CLIP, tileGridSize=CLAHE_TILE_GRID)
        eq = cv2.GaussianBlur(clahe.apply(gray), (3, 3), 0)
        window = int(round(SAUVOLA_WINDOW * scale * SAUVOLA_STRONG_WINDOW_FACTOR))
        binary = _sauvola_threshold(eq, window, SAUVOLA_STRONG_K, SAUVOLA_R)
        return _clean_binary_contours(binary)
    except Exception as e:
        logging.debug(f"[ocr_variant:sauvola_strong] falló: {e}")
        return None


OCR_VARIANT_BUILDERS = {
    "clahe": variant_clahe,
    "clahe_unsharp": variant_clahe_unsharp,
    "sauvola_strong": variant_sauvola_strong,
}
