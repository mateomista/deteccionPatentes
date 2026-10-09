"""Carga de modelos pesados (RF-DETR, RT-DETR, EasyOCR).

Se cargan UNA vez en un `ModelBundle` y se reutilizan entre videos y entre
corridas del pipeline: la calibración y la evaluación corren el pipeline muchas
veces y recargar tres modelos en cada una sería el cuello de botella.

Los imports pesados (torch, transformers, rfdetr, easyocr) están dentro de las
funciones para que el resto del paquete (replay, métricas, splits, tests) se
pueda usar sin GPU ni esos paquetes instalados.
"""
import logging

_WANTED_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


def resolve_vehicle_class_names():
    """Mapa {class_id: nombre} armado preguntándole a la propia librería rfdetr,
    buscando por NOMBRE y nunca por id fijo (hay desfasajes de indexado de a uno
    entre versiones/checkpoints; ver roboflow/rf-detr#330). COCO_CLASSES puede
    ser lista (id = índice) o dict {id: nombre} según la versión."""
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES
    except ImportError:
        from rfdetr.assets.coco_classes import COCO_CLASSES

    pairs = COCO_CLASSES.items() if isinstance(COCO_CLASSES, dict) else enumerate(COCO_CLASSES)
    mapping = {cid: name for cid, name in pairs if name in _WANTED_VEHICLE_NAMES}
    if set(mapping.values()) != _WANTED_VEHICLE_NAMES:
        logging.warning(
            "No se encontraron todas las clases de vehiculo esperadas en COCO_CLASSES "
            f"de rfdetr. Encontradas: {mapping}. Revisá COCO_CLASSES a mano.")
    return mapping


def configure_torch(cudnn_benchmark=False):
    """TF32 en Ampere. [FIX-v5 #6] cudnn.benchmark es opt-in: con EasyOCR las
    shapes cambian en cada crop y el autotuner puede costar más de lo que ahorra."""
    import torch
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()


def load_ocr_reader(gpu=True):
    import easyocr
    logging.info("Cargando EasyOCR...")
    return easyocr.Reader(['en'], gpu=bool(gpu), verbose=False)


class ModelBundle:
    """Modelos cargados una sola vez. `load_vehicle` / `load_plate` en False
    permiten, por ejemplo, evaluar solo el OCR o solo el detector de patentes."""

    def __init__(self, plate_model_path, device=None, cudnn_benchmark=False,
                 load_vehicle=True, load_plate=True, load_ocr=True):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        configure_torch(cudnn_benchmark)
        logging.info(f"Utilizando dispositivo de base: {self.device.upper()}")

        self.vehicle_model = None
        self.vehicle_class_names = {}
        self.plate_processor = None
        self.plate_model = None
        self.ocr_reader = None

        if load_vehicle:
            from rfdetr import RFDETRMedium
            self.vehicle_class_names = resolve_vehicle_class_names()
            logging.info(f"[VEHICLE_CLASS_NAMES] Mapeo real resuelto desde rfdetr: "
                         f"{self.vehicle_class_names}")
            logging.info("Cargando modelo RF-DETR (Vehículos)...")
            self.vehicle_model = RFDETRMedium()
            if self.device == "cuda" and hasattr(self.vehicle_model, "to"):
                self.vehicle_model.to(self.device)
            elif self.device == "cuda":
                logging.info("RF-DETR no expone .to(); el dispositivo real depende de "
                             "su implementación interna.")

        if load_plate:
            load_plate_detector(self, plate_model_path)

        if load_ocr:
            self.ocr_reader = load_ocr_reader(gpu=(self.device == "cuda"))


def load_plate_detector(bundle, path):
    """Carga (o recarga) el RT-DETR de patentes en un bundle existente. Sirve
    para comparar checkpoints sin recargar RF-DETR ni EasyOCR."""
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    logging.info(f"Cargando modelo RT-DETR (Patentes) desde {path} ...")
    bundle.plate_processor = RTDetrImageProcessor.from_pretrained(path)
    bundle.plate_model = RTDetrForObjectDetection.from_pretrained(path).to(bundle.device)
    bundle.plate_model.eval()
