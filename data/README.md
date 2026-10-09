# Datos (no versionados)

```
data/
  videos/                  # <nombre>.mp4  (el nombre sin extensión es la clave del video)
  ground_truth.csv         # verdad-terreno end-to-end, una fila por VEHÍCULO
  CCPD2019/                # imágenes del detector (o tu set propio)
  ocr_calib/               # recortes de patente + labels.csv (calibración del OCR)
  ocr_test/                # recortes de patente + labels.csv (test del OCR)
```

`ground_truth.csv`:

```
video,plate,first_frame,last_frame,vehicle_type
trafico01,AB123CD,120,310,car
trafico01,A123BCD,402,515,motorcycle
```

`first_frame`/`last_frame` = tramo en que ese vehículo es visible en la zona de lectura.
Una detección cuenta para el vehículo si su `frame_idx` cae en ese tramo.

`labels.csv` del OCR: `image,plate,vehicle_class` (`vehicle_class` opcional).
