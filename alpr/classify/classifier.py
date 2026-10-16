"""Clasificador de vehículos (marca y modelo) con Hugging Face Transformers, o modo simulado."""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .labels import parse_brand_model

logger = logging.getLogger("clasificador_offline")


class VehicleClassifier:
    """Clasifica marca y modelo con un modelo de imágenes de Hugging Face Transformers (Apache-2.0).

    Reemplaza a Ultralytics YOLO-cls: Ultralytics se distribuye bajo AGPL-3.0 (copyleft), que obliga a
    publicar el código derivado o a comprar una licencia comercial. Ver docs/LICENCIAS.md.

    El modelo es un DIRECTORIO (igual que el RT-DETR de patentes): `config.json` con `id2label`,
    los pesos y `preprocessor_config.json`. Los nombres de clase deben tener la forma `Marca_Modelo`.
    """

    def __init__(
        self,
        model_path: Path,
        device: str,
        mock: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.mock = mock
        self.model = None
        self.processor = None
        self._torch = None

        if self.mock:
            logger.warning(
                "MODO MOCK ACTIVADO. No se cargará ni utilizará un modelo de red neuronal."
            )
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo de clasificación: {self.model_path}\n"
                "Es un directorio con config.json, los pesos y preprocessor_config.json.\n"
                "Para operar sin modelo entrenado, utilizá el argumento --mock."
            )

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError(
                "Faltan torch o transformers en este entorno.\n"
                "Instalá con: pip install torch transformers"
            ) from exc

        logger.info("Cargando modelo de clasificación desde: %s", self.model_path)
        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        self.model = (
            AutoModelForImageClassification.from_pretrained(str(self.model_path))
            .to(self.device)
            .eval()
        )
        logger.info("Modelo cargado exitosamente en dispositivo: %s", self.device)

    def classify(
        self,
        image_path: Path,
    ) -> tuple[str, Optional[str], Optional[str], float]:
        if self.mock:
            # En modo mock no pretendemos que exista un modelo ni inventamos predicciones reales
            return ("MOCK_VEHICULO", None, None, 0.0)

        if self.model is None or self.processor is None:
            raise RuntimeError("No hay un modelo de clasificación cargado.")

        # np.fromfile + imdecode: funciona con rutas con tildes en Windows (cv2.imread no).
        bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"No se pudo decodificar la imagen: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probs = self._torch.softmax(logits.float(), dim=-1)
            top_prob, top_index = probs.max(dim=-1)
            class_id = int(top_index)
            confidence = float(top_prob)

        label = str(self.model.config.id2label[class_id])
        brand, model = parse_brand_model(label)

        return (label, brand, model, confidence)
