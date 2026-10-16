"""Carga los scripts de standalone/ y legacy/ (que no son paquetes) como módulos de prueba."""
import importlib
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def install_light_fakes_for_missing_heavy_modules():
    """rfdetr y easyocr pueden no estar instalados; las pruebas simulan los modelos y no los usan."""
    try:
        importlib.import_module("rfdetr")
    except ImportError:
        top = types.ModuleType("rfdetr")
        top.RFDETRMedium = object
        util = types.ModuleType("rfdetr.util")
        coco = types.ModuleType("rfdetr.util.coco_classes")
        coco.COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
        sys.modules.update({"rfdetr": top, "rfdetr.util": util, "rfdetr.util.coco_classes": coco})
    try:
        importlib.import_module("easyocr")
    except ImportError:
        sys.modules["easyocr"] = types.ModuleType("easyocr")


def load(name, relpath):
    install_light_fakes_for_missing_heavy_modules()
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_standalone_classifier():
    return load("standalone_classifier", "standalone/pipeclasificadorVehiculos.py")
