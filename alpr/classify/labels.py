"""Catálogo de marcas e interpretación de etiquetas de clasificación (marca, modelo).

Lógica pura: sin OpenCV, sin torch, sin sistema de archivos.
"""
import unicodedata
from typing import Optional

# Marcas compuestas ordenadas primero para evitar colisiones de prefijo
KNOWN_BRANDS = [
    # Marcas compuestas
    "mercedes_benz",
    "mercedes-benz",
    "mercedes benz",
    "alfa_romeo",
    "alfa-romeo",
    "alfa romeo",
    "aston_martin",
    "aston-martin",
    "aston martin",
    "land_rover",
    "land-rover",
    "land rover",
    "rolls_royce",
    "rolls-royce",
    "rolls royce",
    "general_motors",
    "general-motors",
    "general motors",
    # Marcas simples estándar en Argentina y mercado internacional
    "volkswagen",
    "chevrolet",
    "toyota",
    "ford",
    "fiat",
    "renault",
    "peugeot",
    "citroen",
    "honda",
    "nissan",
    "jeep",
    "hyundai",
    "kia",
    "audi",
    "bmw",
    "ram",
    "chery",
    "byd",
    "geely",
    "volvo",
    "suzuki",
    "mitsubishi",
    "subaru",
    "mazda",
    "dodge",
    "chrysler",
    "seat",
    "porsche",
    "ds",
    "iveco",
    "scania",
]

CANONICAL_BRAND_NAMES = {
    "mercedes_benz": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "alfa_romeo": "Alfa Romeo",
    "alfa-romeo": "Alfa Romeo",
    "alfa romeo": "Alfa Romeo",
    "aston_martin": "Aston Martin",
    "aston-martin": "Aston Martin",
    "aston martin": "Aston Martin",
    "land_rover": "Land Rover",
    "land-rover": "Land Rover",
    "land rover": "Land Rover",
    "rolls_royce": "Rolls-Royce",
    "rolls-royce": "Rolls-Royce",
    "rolls royce": "Rolls-Royce",
    "general_motors": "General Motors",
    "general-motors": "General Motors",
    "general motors": "General Motors",
    "volkswagen": "Volkswagen",
    "chevrolet": "Chevrolet",
    "toyota": "Toyota",
    "ford": "Ford",
    "fiat": "Fiat",
    "renault": "Renault",
    "peugeot": "Peugeot",
    "citroen": "Citroen",
    "honda": "Honda",
    "nissan": "Nissan",
    "jeep": "Jeep",
    "hyundai": "Hyundai",
    "kia": "Kia",
    "audi": "Audi",
    "bmw": "BMW",
    "ram": "RAM",
    "chery": "Chery",
    "byd": "BYD",
    "geely": "Geely",
    "volvo": "Volvo",
    "suzuki": "Suzuki",
    "mitsubishi": "Mitsubishi",
    "subaru": "Subaru",
    "mazda": "Mazda",
    "dodge": "Dodge",
    "chrysler": "Chrysler",
    "seat": "SEAT",
    "porsche": "Porsche",
    "ds": "DS",
    "iveco": "Iveco",
    "scania": "Scania",
}


def _fold(text: str) -> str:
    """Minúsculas y sin tildes/diéresis, para comparar con el catálogo ('Citroën' == 'citroen')."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _canon_key(brand: str) -> str:
    return _fold(brand).replace("-", "_").replace(" ", "_")


def parse_brand_model(label: str) -> tuple[Optional[str], Optional[str]]:
    """
    Interpreta y descompone etiquetas de clasificación en tuplas (marca, modelo).
    
    Formatos y reglas soportadas:
    1. Separadores explícitos de alta prioridad:
       - 'Toyota::Corolla' -> ('Toyota', 'Corolla')
       - 'Mercedes-Benz|C200' -> ('Mercedes-Benz', 'C200')
       - 'Ford/F_150' -> ('Ford', 'F_150')
    2. Marcas del catálogo reconocidas (incluyendo compuestas):
       - 'Ford_F_150' -> ('Ford', 'F_150')
       - 'Volkswagen_Gol_Trend' -> ('Volkswagen', 'Gol_Trend')
       - 'Mercedes_Benz_C200' -> ('Mercedes-Benz', 'C200')
       - 'Alfa_Romeo_Giulia' -> ('Alfa Romeo', 'Giulia')
    3. Marcas simples no catalogadas con exactamente un separador:
       - 'MarcaX_ModeloY' -> ('MarcaX', 'ModeloY')
       
    Casos ambiguos (marcas desconocidas con múltiples guiones) o etiquetas vacías
    devuelven (None, None) de manera controlada sin arrojar excepciones.
    """
    if not label or not isinstance(label, str):
        return None, None

    # NFC: cada letra con tilde es UN carácter, así el largo coincide con la versión sin tildes.
    raw = unicodedata.normalize("NFC", label.strip())
    if not raw:
        return None, None

    # 1. Separadores explícitos inequívocos
    for sep in ("::", "|", "/"):
        if sep in raw:
            parts = raw.split(sep, 1)
            b = parts[0].strip() or None
            m = parts[1].strip() or None
            if b:
                b_canon = CANONICAL_BRAND_NAMES.get(_canon_key(b), b)
            else:
                b_canon = None
            return b_canon, m

    raw_lower = _fold(raw)

    # 2. Búsqueda contra catálogo canónico (marcas compuestas primero)
    for brand_key in KNOWN_BRANDS:
        for delimiter in ("_", "-", " "):
            prefix = brand_key + delimiter
            if raw_lower.startswith(prefix):
                brand_canonical = CANONICAL_BRAND_NAMES.get(brand_key, brand_key.title())
                model_part = raw[len(prefix):].strip()
                return brand_canonical, (model_part or None)

    # 2b. La etiqueta ES una marca del catálogo (clasificador por marca): sin modelo.
    #     Debe ir antes del fallback, que partiría 'Mercedes-Benz' en ('Mercedes', 'Benz').
    bare = CANONICAL_BRAND_NAMES.get(_canon_key(raw))
    if bare is not None:
        return bare, None

    # 3. Fallback: exactamente un separador ('_' o '-')
    for delimiter in ("_", "-"):
        if raw.count(delimiter) == 1:
            b, m = raw.split(delimiter, 1)
            b = b.strip() or None
            m = m.strip() or None
            if b:
                b_canon = CANONICAL_BRAND_NAMES.get(_canon_key(b), b)
            else:
                b_canon = None
            return b_canon, m

    # 4. Caso no separable o ambiguo
    return None, None
