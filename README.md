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
  contracts.py      contrato productor-consumidor (nombres de archivo, escritura atómica)
  classify/         clasificador offline de vehículos (marca/modelo), consumidor desacoplado
  tools/            cut_video.py · compare_runs.py (V5 vs modular) · join_results.py (patente + clasificación)
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

## Clasificador de vehículos (`alpr.classify`)

Consumidor desacoplado que clasifica la marca y el modelo de cada recorte de vehículo
(`vehiculo_<track_id>.<ext>`) de una carpeta. Es el script `pipeclasificadorVehiculos.py`
reorganizado en módulos; el original sigue intacto en `legacy/`.

```
alpr/classify/
  config.py       constantes (carpetas, tiempos, reintentos)
  labels.py       catálogo de marcas y parser de etiquetas (lógica pura)
  files.py        candidatos, hash SHA-256 y espera de archivo estable (Windows)
  results.py      ClassificationResult y escritor JSONL
  classifier.py   YOLO-cls o modo --mock
  watchers.py     watchdog (opcional) o sondeo periódico
  processor.py    cola + worker idempotente
  cli.py          python -m alpr.classify
```

```bash
python -m alpr.classify --help
python -m alpr.classify --mock --single-run --device cpu --input-dir <carpeta> --results-file <salida.jsonl>
python -m pytest tests/classify -q
```

`watchdog` y `ultralytics` son opcionales y NO se instalan con `requirements.txt`
(ver `requirements-classifier.txt`; instalar solo con autorización del equipo).

### Contrato con el pipeline (productor)

El pipeline puede exportar el recorte del vehículo de cada patente registrada, listo para el clasificador:

```bash
python -m alpr.inference --source clip.mp4 --strict-plate --export-vehicle-crops outputs/crops_vehiculos
python -m alpr.classify --mock --single-run --device cpu
python -m alpr.tools.join_results --detections detecciones_alpr.json --classifications outputs/clasificaciones_vehiculos.jsonl --video clip
```

- **Nombre:** `vehiculo_<track_id>__<video>.jpg`. El video va en el nombre porque el `track_id` de ByteTrack se
  reinicia en cada video (sin él, `vehiculo_1.jpg` de dos videos colisionaría). El formato original
  `vehiculo_<track_id>.jpg` sigue aceptándose.
- **Escritura atómica** (`.tmp` y renombrar): el consumidor nunca ve un archivo a medio escribir.
- **Apagado por defecto:** sin `--export-vehicle-crops` el pipeline se comporta como antes; con la opción, las
  detecciones son las mismas (lo verifican las pruebas).
- **Qué se exporta:** un recorte por patente registrada, el de la lectura que la respalda. Los vehículos sin patente
  leída no se exportan todavía.
- **Memoria:** con la opción encendida, cada patente candidata conserva una copia del recorte del vehículo hasta
  que el track termina.
- **GPU:** el productor y el consumidor son procesos separados que se comunican por una carpeta: el clasificador no
  puede frenar al pipeline y se puede correr en CPU (`--device cpu`). Cuánto cuesta clasificar en GPU junto al
  pipeline hay que medirlo cuando exista el modelo.

Para quien prefiera seguir con los dos scripts sueltos, las mismas mejoras están en `standalone/`
(ver `standalone/README.md`).

### Qué cambió respecto del original

La reorganización conserva el código: cada clase y función es idéntica salvo `main`, que ahora
configura el logging al arrancar y no al importar. Además se corrigieron estos problemas, cada uno
con una prueba que **fallaba sobre el código original** (`tests/classify/`):

| Problema en el original | Corrección |
|---|---|
| Un recorte probado con `--mock` contaba como "ya procesado" para el modelo real | El modo real ignora los registros `mock` al cargar los hashes |
| Ctrl+C mandaba a `fallidos/` los recortes que no llegaron a procesarse | Con cierre solicitado quedan en la carpeta de entrada para el próximo arranque |
| Si fallaba la escritura del JSONL después de mover el archivo, el registro se perdía en silencio | La escritura se reintenta y, si no se logra, el registro completo queda en el log |
| Una línea JSONL cortada por una caída absorbía el registro siguiente | Se cierra la línea parcial antes de anexar |
| La etiqueta `Mercedes-Benz` sola se partía en marca `Mercedes` y modelo `Benz`; una etiqueta de solo marca (`toyota`) daba `(None, None)` | Una etiqueta que es una marca del catálogo devuelve `(marca, None)` |
| `Citroën_C3` y `citroen_c3` daban marcas distintas | La comparación con el catálogo ignora tildes |
| La espera de archivo estable usaba `cv2.imread`, que falla en Windows con rutas con tildes o ñ | Se lee con `np.fromfile` + `imdecode` (no verificado en Windows) |

### Limitaciones conocidas (no se tocaron)

- **Sin patente leída no hay recorte.** El productor exporta un recorte por patente registrada; los vehículos
  cuya patente no se leyó no se clasifican todavía.
- **Sin modelo ni dataset.** No hay `weights/vehicle_classifier.pt`; solo funciona `--mock`.
- **Modo watchdog sin reescaneo.** Si la cola (1000) se llena, un archivo no encolado se recupera
  recién en el próximo arranque; el sondeo periódico sí lo recupera.
- **Umbral de confianza en 0,0:** el estado `below_threshold` nunca se produce, y no hay un flag para cambiarlo.
- **Rutas con tildes en la clasificación real:** la lectura de la espera es segura, pero no se verificó
  cómo carga `ultralytics` las imágenes con esas rutas en Windows.

## Comparar contra V5 (paridad)

Antes de dar por buena la migración, correr los dos sobre el MISMO clip y comparar:

```bash
python legacy/pipelineV5.py --source clip.mp4 --plate-model outputs/final_model --strict-plate --output-json outputs/v5.json
python -m alpr.inference    --source clip.mp4 --plate-model outputs/final_model --strict-plate --output-json outputs/modular.json
python -m alpr.tools.compare_runs --a outputs/v5.json --b outputs/modular.json --label-a V5 --label-b modular
```

Clasifica cada detección como coincidencia, lectura distinta del mismo vehículo, o presente en una sola
corrida, y sale con código 0 solo si hay paridad (desfase de frames <= `--frame-tolerance`, 5 por defecto).
Acepta varios videos a la vez (`--a a1.json a2.json --b b1.json b2.json`, emparejados por orden).

## Qué descarta el pipeline (y por qué)

Cada corrida con `--output-json X.json` escribe además `X_descartes.json`, y los recortes descartados
quedan en `snapshots/descartes/` (si hay `--snapshots-dir`). Motivos:

| `motivo` | Qué pasó |
|---|---|
| `rtdetr_sin_patente` / `caja_de_patente_invalida` | RT-DETR no encontró patente en el vehículo |
| `focus_mask` | Se encontró una caja pero se descartó por tamaño, aspecto o contraste (`detalle`) |
| `ocr_sin_lectura_valida` | Los intentos de OCR no dieron una lectura válida; `intentos` muestra qué leyó cada variante (`texto_leido`, `texto_corregido`, `confianza`, `resultado`) |
| `vehiculo_nunca_cruzo_zona_ocr` | El vehículo nunca llegó a la zona de lectura |
| `vehiculo_sin_lecturas_validas` | Entró a la zona pero ninguna lectura fue válida |
| `vehiculo_sin_consenso_no_recuperable` | Hubo lecturas pero no alcanzaron para consenso ni para recuperación (`lecturas` las detalla) |
| `patente_duplicada` / `registro_rechazado` | Se descartó por deduplicación o por evidencia inconsistente |

El log final resume los motivos y explica vehículo por vehículo por qué no salió patente.

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
