"""Clasificador offline de vehículos (consumidor desacoplado del pipeline de patentes).

Arquitectura productor-consumidor: el pipeline escribe recortes de vehículos
(`vehiculo_<track_id>.<ext>`) en una carpeta y este consumidor los clasifica de
forma idempotente. Uso: `python -m alpr.classify --help`.
"""
