# legacy/ — originales intactos

Copias SIN MODIFICAR de los scripts de partida, conservadas como respaldo y como referencia
para comparar el comportamiento del paquete `alpr/`. No se editan: si hay que cambiar algo,
se cambia en `alpr/`.

| Archivo | Reemplazado por |
|---|---|
| `pipelineV5.py` | `alpr/` (pipeline, entrenamiento, calibración, evaluación) |
| `pipeclasificadorVehiculos.py` | `alpr/classify/` |

**Aviso de licencia:** `pipeclasificadorVehiculos.py` usa Ultralytics (AGPL-3.0). No usarlo en producción; ver
`docs/LICENCIAS.md`. Las versiones sin Ultralytics son `alpr/classify/` y `standalone/pipeclasificadorVehiculos.py`.

`SHA256SUMS.txt` registra el hash de cada original. Para verificar que nadie los modificó:

```bash
cd legacy && sha256sum -c SHA256SUMS.txt        # Linux / Git Bash
```
```powershell
Get-FileHash legacy\*.py -Algorithm SHA256       # PowerShell: comparar con SHA256SUMS.txt
```
