"""Clasificador de vehículos con YOLO-cls (pesos reales o modo simulado)."""
import logging
from pathlib import Path
from typing import Optional

from .labels import parse_brand_model

logger = logging.getLogger("clasificador_offline")


class VehicleClassifier:
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

        if self.mock:
            logger.warning(
                "MODO MOCK ACTIVADO. No se cargará ni utilizará un modelo de red neuronal."
            )
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo de modelo: {self.model_path}\n"
                "Para operar sin pesos entrenados, utilizá el argumento --mock."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics no está instalado en este entorno.\n"
                "Instalá con: pip install ultralytics"
            ) from exc

        logger.info("Cargando modelo YOLO-cls desde: %s", self.model_path)
        self.model = YOLO(str(self.model_path))
        logger.info("Modelo cargado exitosamente en dispositivo: %s", self.device)

    def classify(
        self,
        image_path: Path,
    ) -> tuple[str, Optional[str], Optional[str], float]:
        if self.mock:
            # En modo mock no pretendemos que exista un modelo ni inventamos predicciones reales
            return ("MOCK_VEHICULO", None, None, 0.0)

        if self.model is None:
            raise RuntimeError("No hay un modelo de clasificación cargado.")

        results = self.model.predict(
            source=str(image_path),
            device=self.device,
            verbose=False,
        )

        if not results:
            raise RuntimeError("El modelo YOLO no devolvió resultados de inferencia.")

        result = results[0]
        if result.probs is None:
            raise RuntimeError(
                "El modelo no posee salidas de clasificación probabilística (YOLO-cls requerido)."
            )

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        label = str(result.names[class_id])
        brand, model = parse_brand_model(label)

        return (label, brand, model, confidence)
