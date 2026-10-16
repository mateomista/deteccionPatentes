# standalone/ — las mejoras, sin modularizar

Para quien prefiera seguir con los dos scripts sueltos. Son **copias** de `legacy/` con las mismas
mejoras que tiene el paquete `alpr/`; los originales de `legacy/` no se tocaron.

| Archivo | Parte de | Qué cambia | Tamaño del cambio |
|---|---|---|---|
| `pipelineV5.py` | `legacy/pipelineV5.py` | Exporta el recorte del vehículo de cada patente registrada (`--export-vehicle-crops`, **apagado por defecto**) | `pipelineV5.patch`: +100 −3 |
| `pipeclasificadorVehiculos.py` | `legacy/pipeclasificadorVehiculos.py` | 7 correcciones + nombre de archivo con video + campo `video` + **sin Ultralytics (AGPL-3.0)**, usa Transformers (Apache-2.0) | `pipeclasificadorVehiculos.patch` |
| `join_results.py` | (nuevo) | Une detecciones y clasificaciones por (video, track_id). Solo biblioteca estándar | — |

Los `.patch` son los diffs exactos contra `legacy/` (para revisar en una pasada). `SHA256SUMS.txt` registra
el hash de cada archivo de esta carpeta, igual que `legacy/SHA256SUMS.txt` para los originales.

## Uso

```bash
# 1) Productor: V5 normal + exportación de recortes de vehículo
python standalone/pipelineV5.py --source clip.mp4 --plate-model outputs/final_model --strict-plate \
    --output-json outputs/clip.json --export-vehicle-crops outputs/crops_vehiculos

# 2) Consumidor: clasifica lo que el productor dejó (en modo simulado mientras no haya modelo)
python standalone/pipeclasificadorVehiculos.py --mock --single-run --device cpu

# 3) Unir patente + clasificación
python standalone/join_results.py --detections outputs/clip.json \
    --classifications outputs/clasificaciones_vehiculos.jsonl --video clip --out outputs/clip_unido.json
```

Sin `--export-vehicle-crops`, `standalone/pipelineV5.py` produce **exactamente** lo mismo que el original
(lo verifica una prueba que corre los dos sobre el mismo video).

## Los dos caminos son equivalentes

Las pruebas de `tests/standalone/` comprueban, con modelos simulados, que:

- las 32 pruebas del clasificador pasan igual sobre el script suelto y sobre `alpr.classify`;
- V5 con mejoras (exportación apagada) da el mismo resultado que el V5 original;
- V5 con mejoras y el pipeline modular dan las mismas detecciones y exportan el mismo archivo;
- el contrato de nombres es idéntico en las cuatro copias (`alpr/contracts.py`, V5, clasificador, `join_results`).

## Qué se paga por no modularizar

- **El contrato está copiado en cuatro lugares.** Si cambia, hay que cambiarlo en todos; las pruebas de
  arriba avisan si divergen, pero no lo corrigen solas.
- **V5 sigue siendo un archivo de 2.376 líneas** con sus limitaciones (umbrales globales, sin calibración ni
  evaluación, sin pruebas unitarias). Esta carpeta agrega la exportación; no resuelve eso.
- **Dos scripts que se invocan por separado**, sin una definición común de configuración.

## Contrato productor-consumidor

Ver `alpr/contracts.py` (misma lógica en todas las copias):

- Nombre: `vehiculo_<track_id>__<video>.jpg`. El video va en el nombre porque el `track_id` de ByteTrack se
  reinicia en cada video. El formato original `vehiculo_<track_id>.jpg` sigue siendo válido.
- Escritura atómica: se escribe `<nombre>.tmp` y se renombra; el consumidor ignora los `.tmp`.
- Qué se exporta: un recorte por cada patente **registrada** (consenso o recuperada), el del frame de la lectura
  que respalda esa patente. Los vehículos sin patente leída **no** se exportan todavía.
- Clave de correlación: (video, `id_vehiculo` de la detección) = (video, `track_id` de la clasificación).
