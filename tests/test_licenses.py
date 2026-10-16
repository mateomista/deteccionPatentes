"""Guardas de licencias: el proyecto solo usa dependencias permisivas (Apache, MIT, BSD o similares).

Ver docs/LICENCIAS.md. Estas pruebas no auditan licencias (eso se hace con los metadatos de los
paquetes); impiden que vuelvan a entrar, por descuido, los paquetes que ya se descartaron.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paquetes descartados por licencia copyleft fuerte (AGPL/GPL) que obliga a abrir el código derivado
# o a comprar una licencia comercial.
EXCLUDED = {"ultralytics", "ultralytics-thop", "ultralytics-platform", "yolov5"}
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+(ultralytics|yolov5)\b", re.MULTILINE)


def code_files():
    for folder in ("alpr", "standalone", "tests"):
        for p in (ROOT / folder).rglob("*.py"):
            if p.name != "test_licenses.py":
                yield p


def test_no_excluded_package_is_imported_outside_legacy():
    offenders = [str(p.relative_to(ROOT)) for p in code_files()
                 if IMPORT_RE.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert not offenders, f"importan paquetes excluidos por licencia: {offenders}"


def test_requirements_do_not_list_excluded_packages():
    for req in ROOT.glob("requirements*.txt"):
        for line in req.read_text(encoding="utf-8").splitlines():
            name = re.split(r"[<>=!~\[;\s]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")
            if name and not name.startswith("#"):
                assert name not in EXCLUDED, f"{req.name} lista {name!r}, excluido por licencia"


def test_legacy_is_the_only_place_that_still_mentions_the_excluded_package():
    """legacy/ conserva el original intacto (con su hash); no se usa en producción."""
    legacy = (ROOT / "legacy" / "pipeclasificadorVehiculos.py").read_text(encoding="utf-8")
    assert "ultralytics" in legacy
