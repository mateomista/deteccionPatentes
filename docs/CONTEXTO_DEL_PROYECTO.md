# Contexto del proyecto (para retomar sin depender de ninguna conversación)

Este archivo reúne el contexto y las decisiones del proyecto, para que cualquier persona o herramienta
(un asistente de IA incluido) pueda retomar el trabajo leyendo el repositorio, sin acceso a un chat anterior.
Mantenerlo actualizado: lo que no está acá ni en el código se pierde.

## Qué es
Sistema ALPR/LPR (UTN-GIBD y Jefatura Departamental de Policía): detecta vehículos, lee patentes y
clasifica marca y modelo. Hardware de desarrollo: Windows, GPU NVIDIA RTX 5070 (12 GB). Cámara fija, vista
trasera en tres cuartos, 1920×1080.

## Cómo está organizado
- `alpr/` paquete modular (pipeline, OCR, consenso, clasificador, entrenamiento, calibración, evaluación).
- `legacy/` los scripts originales, SIN modificar, con su hash (`SHA256SUMS.txt`). No se editan.
- `standalone/` las mismas mejoras aplicadas a los dos scripts sueltos, por si el grupo prefiere no modularizar.
- `docs/LICENCIAS.md` auditoría de licencias. `docs/reportes/` reportes de avance diarios.
- `README.md` flujo de trabajo y comandos.

## Criterios de trabajo (acordados)
1. **Licencias:** solo Apache-2.0, MIT, BSD o similares, que permitan comercializar sin pagar ni abrir el
   código propio. Ultralytics (AGPL-3.0) está excluido. Toda dependencia, modelo o dataset nuevo se registra
   en `docs/LICENCIAS.md` con la licencia que declara su ficha.
2. **No tocar `legacy/`.** Los cambios van en `alpr/` o en `standalone/`.
3. **Cambios incrementales y verificables:** cada corrección con su prueba (`python -m pytest -q`), y no dar
   una fase por terminada solo porque se informe que está implementada: mostrar comandos y resultados reales.
4. **Nada se instala sin autorización del grupo** (dependencias opcionales como `watchdog`).
5. **Números citables:** calibrar en `calib`, evaluar una sola vez en `test`, particiones por video o vehículo,
   nunca por frame.

## Decisiones tomadas
- El video va en el nombre del recorte exportado (`vehiculo_<track_id>__<video>.jpg`) porque el `track_id` se
  reinicia en cada video. Escritura atómica. Ver `alpr/contracts.py`.
- La exportación de recortes de vehículo está apagada por defecto; con ella encendida las detecciones no cambian.
- El clasificador usa Transformers (Apache-2.0). Las clases se nombran `Marca_Modelo`.
- Los pesos solo se promueven a `weights/` si cumplen el criterio de aceptación en `test`.
- Los AVI exportados por el sistema de video declaran 120 fps pero son de ~23 fps variable: siempre pasarlos por
  `python -m alpr.tools.cut_video`, que corta por el reloj del video y escribe el fps real.

## Estado
Ver el último reporte en `docs/reportes/`. Lo pendiente principal: modelo y dataset de marca y modelo, repetir la
comprobación de paridad con V5 en más clips, ground truth etiquetado, tapar el texto de la cámara en los recortes,
y las decisiones de licencia sobre componentes LGPL y NVIDIA.

## Cosas que se aprendieron y conviene no repetir
- Python 3.15 no sirve (falta `scikit-image` precompilado): usar 3.12.
- Una GPU RTX 50xx necesita `torch` con CUDA 12.8 o superior.
- `cv2.imread`/`VideoCapture` fallan con rutas con tildes en Windows: copiar a un nombre simple o usar
  `np.fromfile` + `imdecode`.
- En PowerShell se pega un comando por vez (las líneas pegadas juntas se fusionan).
- Git en Windows puede convertir finales de línea y romper los hashes: `.gitattributes` lo evita en `legacy/` y `standalone/`.
