# ALPR — entrenamiento, calibración y evaluación separados

Pipeline de lectura de patentes (RF-DETR → ByteTrack → RT-DETR → EasyOCR → consenso)
partido en bloques que se entrenan, calibran y evalúan por separado. Es el refactor de
`pipelineV5.py`: la lógica de detección/OCR/consenso/exportación es la misma; cambia cómo
está organizada y cómo se alimenta.

## Qué se entrena y qué se calibra

| Pieza | ¿Cómo se ajusta? | Dónde |
|---|---|---|
| RF-DETR (vehículos), EasyOCR | Preentrenados, no se tocan | — |
| RT-DETR (patentes) | **Entrenamiento** (fine-tuning) | `alpr.train.train_rtdetr` |
| 8 umbrales `[SIN CALIBRAR]` | **Calibración** contra verdad-terreno | `alpr.calibrate.calibrate` |
| Todo, con umbrales congelados | **Evaluación** en test (una vez) | `alpr.evaluate.*` |

## Particiones (sin fuga de datos)

Siempre se parte por **video / secuencia**, nunca por frame.

| Partición | Para qué | Se puede mirar |
|---|---|---|
| `train` | entrenar RT-DETR | siempre |
| `val` | elegir checkpoint y `confidence_plate` | siempre |
| `calib` | barrer umbrales y variantes OCR | siempre |
| `test` | número final de la tesis | **una vez**, con todo congelado |

`calibrate record` se niega a correr sobre videos de `test`.

## Estructura

```
alpr/
  config.py         constantes de algoritmo + Thresholds (calibrables) + PipelineOptions
  imaging.py        prewarp, focus mask, Sauvola, limpieza, variantes CLAHE/unsharp   (sin GPU)
  plate_text.py     validación, corrección, fusión de fragmentos                       (sin GPU)
  tracking.py       VehicleTrack: votos, consenso, recuperación                        (sin GPU)
  finalize.py       armado + auditoría de registros (compartido pipeline/replay)       (sin GPU)
  ocr.py            PlateOCR: OCR adaptativo de 1 crop (evaluable solo)
  models.py         ModelBundle: carga RF-DETR/RT-DETR/EasyOCR UNA vez
  pipeline.py       ALPRPipeline.run_video(): reutilizable, devuelve RunResult
  replay.py         re-decide el consenso OFFLINE sobre lecturas grabadas              (sin GPU)
  inference.py      CLI de producción
  data/             ccpd.py (CCPD→COCO) · splits.py (particiones sin fuga)
  train/            train_rtdetr.py
  calibrate/        calibrate.py  (record → sweep)
  tools/            cut_video.py (cortar un tramo de un video sin ffmpeg)
  evaluate/         e2e.py · ocr_eval.py · detector_eval.py · metrics.py · detection_metrics.py
legacy/            pipelineV5.py original, para comparar salidas contra el refactor
tests/              56 tests: lógica pura + pipeline real con modelos simulados
```

## Flujo completo

```bash
pip install -r requirements.txt

# 0) Datos del detector: CCPD -> COCO -> particiones train/val/test
python -m alpr.data.ccpd   --images-dir data/CCPD2019 --out data/ccpd_coco.json
python -m alpr.data.splits coco --annotations data/ccpd_coco.json \
    --split train=0.8 --split val=0.1 --split test=0.1 --out-dir data/detector

# 1) ENTRENAR el detector de patentes (usa train + val)
python -m alpr.train.train_rtdetr \
    --train-ann data/detector/train.json --train-images data/CCPD2019 \
    --val-ann   data/detector/val.json   --val-images   data/CCPD2019

# 1b) Evaluar el detector: en val para elegir confidence_plate, en test una vez
python -m alpr.evaluate.detector_eval --model outputs/rtdetr-ccpd/final_model \
    --annotations data/detector/val.json --images-dir data/CCPD2019

# 2) Particionar los VIDEOS etiquetados (data/ground_truth.csv) en calib / test
python -m alpr.data.splits videos --videos-dir data/videos \
    --ground-truth data/ground_truth.csv --split calib=0.5 --split test=0.5

# 3) CALIBRAR
#    3a) grabar lecturas (GPU, una vez por video y por combinación de umbrales de detección)
python -m alpr.calibrate.calibrate record --split calib --strict-plate \
    --confidence-vehicle-grid 0.35 0.45 --confidence-plate-grid 0.15 0.25 0.4
#    3b) barrer umbrales de decisión OFFLINE (segundos, sin GPU)
python -m alpr.calibrate.calibrate sweep --objective recall_at_precision --min-precision 0.95
#    (opcional) A/B del OCR sobre recortes etiquetados de calibración
python -m alpr.evaluate.ocr_eval --labels data/ocr_calib/labels.csv --crops-dir data/ocr_calib --strict-plate

# 4) EVALUAR en test, UNA vez, umbrales congelados (corre el pipeline real, no el replay)
python -m alpr.evaluate.e2e --split test --strict-plate \
    --thresholds outputs/calibration/best_thresholds.json
# línea base sin calibrar, para mostrar cuánto aportó la calibración:
python -m alpr.evaluate.e2e --split test --strict-plate --default-thresholds --out-dir outputs/eval_test_default

# 5) Producción
python -m alpr.inference --source inputs/trafico.mp4 --strict-plate \
    --thresholds outputs/calibration/best_thresholds.json
```

## Por qué la calibración es rápida

Los umbrales son de dos clases:

- **Detección** (`confidence_vehicle`, `confidence_plate`): cambian qué ven los modelos → hay
  que volver a correr la GPU por cada valor. Se barren en `record`.
- **Decisión** (`min_votes`, `min_avg_ocr_conf`, `early_finalize_*`, `min_consensus_margin`,
  `flush_single_vote_conf`): cambian solo cómo se decide con las lecturas ya hechas. `record`
  graba cada lectura y `sweep` re-decide con `alpr.replay` en milisegundos: la grilla por
  defecto son 1620 combinaciones.

`replay` usa el mismo `VehicleTrack` y `prepare_record` que el pipeline real, pero es una
aproximación (ver `alpr/replay.py`): por eso **lo que se cita sale de `evaluate.e2e`**, no del
barrido.

## Lo que hay que saber antes de citar números

- Las métricas del `sweep` son sobre `calib` (optimistas). El número de tesis es el de `test`.
- Con pocos vehículos de GT la grilla se sobreajusta; `sweep` avisa si hay menos de 50.
- `--strict-plate` en toda calibración/evaluación: el modo permisivo no equivale a formato argentino.
- Las detecciones `track_lost` / `end_of_video` tienen evidencia más débil; `e2e` las reporta aparte
  ("todas" vs "solo consenso").
- CCPD son patentes chinas: sirve para entrenar el detector, no para calibrar sobre patentes argentinas.
- `_maybe_activate_ocr` (cruce de la zona OCR por el centro del bbox) es una reconstrucción de V3
  señalada en el código original; verificala contra tu V3.

## Tests

```bash
pip install -r requirements-dev.txt && pytest
```

## Qué está verificado y qué no

- **Verificado con tests**: validación/corrección de texto, consenso, recuperación, replay, métricas,
  particiones sin fuga, conversión CCPD, calibración end-to-end (`sweep`), y `ALPRPipeline.run_video`
  completo sobre un video sintético con RF-DETR/RT-DETR/EasyOCR simulados — incluyendo que el
  **replay offline reproduce exactamente lo que decide la corrida real** en 3 configuraciones de umbrales.
- **Verificado con un modelo de juguete en CPU**: el ciclo de `train_rtdetr` (dataset → entrenamiento →
  validación → guardado → recarga con 1 clase) y `detector_eval`.
- **NO verificado (requiere tus pesos, video y GPU)**: la corrida con RF-DETR, el RT-DETR real y EasyOCR;
  el rendimiento (FPS, VRAM); y la calidad del entrenamiento sobre CCPD. Antes de calibrar, corré
  `python -m alpr.inference` sobre un video y compará el JSON con el que te daba `pipelineV5.py`.
