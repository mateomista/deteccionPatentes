# Licencias de las dependencias

**Criterio del proyecto:** solo componentes con licencia Apache-2.0, MIT, BSD o similar, que permitan
desarrollar y comercializar sin pagar licencias ni abrir el código propio.

**Auditoría del 16/10/2026.** Se resolvieron las dependencias de `requirements.txt` y de
`requirements-classifier.txt` con todas sus dependencias transitivas (99 paquetes) y se leyó la licencia que
declara cada paquete en sus metadatos. Es la licencia que declaran los autores en PyPI: **no es asesoramiento
legal**, y no cubre los modelos preentrenados ni los datasets (ver el final).

## Resultado

| Categoría | Paquetes | Estado |
|---|---|---|
| Permisivas (Apache-2.0, MIT, BSD, PSF, HPND y similares) | 80 | Cumplen |
| **AGPL-3.0**: `ultralytics`, `ultralytics-thop`, `ultralytics-platform` | 3 | **Excluidas: no cumplen. Eliminadas del proyecto** |
| LGPL: `python-bidi` (dependencia de EasyOCR) | 1 | Condicional |
| NVIDIA CUDA (se instalan con `torch` para GPU) | 15 | Condicional |
| Total | 99 | |

Sin Ultralytics, el proyecto queda en 96 paquetes.

### Excluido: Ultralytics (AGPL-3.0)

`ultralytics` 8.4.178 declara `AGPL-3.0`. Es una licencia de copyleft fuerte: usarlo en un servicio o
distribuirlo obliga a publicar el código derivado bajo la misma licencia, o a comprar una licencia comercial
(la empresa ofrece una). No cumple el criterio. Se reemplazó por clasificación con Hugging Face Transformers
(Apache-2.0) en `alpr/classify/` y en `standalone/pipeclasificadorVehiculos.py`. Una prueba
(`tests/test_licenses.py`) impide que vuelva a importarse o a listarse en los requirements.
`legacy/pipeclasificadorVehiculos.py` conserva el original, que sí usa Ultralytics: no usarlo en producción.

### Condicionales (LGPL, no son Apache/MIT)

La LGPL permite uso comercial, con condiciones (por ejemplo, avisos de licencia y permitir reemplazar la
biblioteca). No es una licencia permisiva como las otras: **conviene que el grupo decida si la acepta**.

- **`python-bidi` 0.6.11 (LGPL)**: lo trae EasyOCR, que es parte del OCR del proyecto.
- **FFmpeg dentro de OpenCV**: los paquetes `opencv-python` y `opencv-python-headless` redistribuyen FFmpeg,
  que es LGPL (el propio archivo de avisos de terceros del paquete lo indica). El código de OpenCV es Apache-2.0.
  Las versiones no headless para Linux y macOS además incluyen Qt 5 (también LGPL); en Windows no.

### Condicionales (NVIDIA CUDA)

Los 15 paquetes de NVIDIA que instala `torch` con soporte de GPU (cuDNN, cuBLAS, etc.) son software propietario
de NVIDIA bajo su propio contrato de licencia, no Apache/MIT. Los que declaran licencia dicen
`NVIDIA Proprietary`; el resto no la declara. Es inevitable en cualquier despliegue con GPU NVIDIA: revisar el
contrato de NVIDIA si el producto se distribuye.

### Permisivas: las que usa el proyecto directamente

| Paquete | Licencia declarada |
|---|---|
| `torch` | Apache-2.0 y BSD (varias) |
| `transformers`, `accelerate`, `huggingface_hub`, `safetensors` | Apache-2.0 |
| `rfdetr` (código) | Apache-2.0 |
| `easyocr`, `watchdog`, `timm` | Apache-2.0 |
| `opencv-python` (código) | Apache-2.0 |
| `supervision` | MIT |
| `numpy`, `scipy` | BSD-3-Clause |
| `Pillow` | MIT-CMU |

Nota sobre `scipy` y `matplotlib`: su licencia principal es permisiva (BSD y PSF), pero sus archivos de
licencia incluyen avisos de componentes de terceros; no se revisaron uno por uno.

## Modelos y datasets (no cubiertos por la auditoría de paquetes)

Las licencias de los pesos y de los datos son independientes de las del código:

| Elemento | Estado |
|---|---|
| RF-DETR Medium (detector de vehículos) | **Apache-2.0** (verificado en la descripción del paquete `rfdetr`). Solo RF-DETR-XL y 2XL usan otra licencia (PML 1.0): no se usan |
| RT-DETR base `PekingU/rtdetr_r50vd` (detector de patentes) | **Sin verificar**: no se pudo consultar Hugging Face desde el entorno de la auditoría. Revisar la ficha del modelo |
| Pesos de EasyOCR | **Sin verificar** |
| CCPD (dataset del detector de patentes) | **Sin verificar**. Además son patentes chinas |
| Datasets de marca y modelo | **Pendiente de elegir**: verificar licencia y permiso de uso comercial antes de entrenar |
| Modelo base del clasificador (por entrenar) | **Pendiente**: elegir uno con licencia Apache-2.0 o MIT declarada en su ficha de Hugging Face, y registrarla acá |

## Cómo repetir la auditoría

```bash
pip install --dry-run --ignore-installed --quiet --report licencias.json -r requirements.txt -r requirements-classifier.txt
```

El informe lista todos los paquetes con sus metadatos. Algunos paquetes no muestran la licencia en ese
informe (aparece vacía); para esos hay que leer el archivo `METADATA` del paquete (`pip download --no-deps <paquete>`).
Al agregar una dependencia nueva, registrar su licencia acá.
