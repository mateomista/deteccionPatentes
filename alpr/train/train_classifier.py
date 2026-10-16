"""ENTRENAMIENTO del clasificador de marca y modelo de vehículos (Hugging Face Transformers).

    python -m alpr.train.train_classifier \
        --data-dir data/clasificador \
        --base-model <modelo base con licencia permisiva> --base-model-license apache-2.0 \
        --dataset-license "CC BY 4.0" \
        --output-dir outputs/clasificador_candidato \
        --min-accuracy 0.90 --promote-to weights/vehicle_classifier

Dataset: <data-dir>/train/<clase>/*.jpg, <data-dir>/val/<clase>/*.jpg y, para evaluar y promover,
<data-dir>/test/<clase>/*.jpg. Los nombres de clase son `Marca_Modelo` (los lee alpr.classify).
Para repartir un dataset sin mezclar vehículos entre particiones: `python -m alpr.data.splits classes`.

Lo que hace, en este orden:
  1. Exige declarar las licencias del modelo base y del dataset, y solo acepta permisivas
     (Apache-2.0, MIT, BSD, CC BY 4.0...). Queda registrado junto a los pesos.
  2. Busca FUGA de datos entre particiones (copias exactas o casi idénticas) y se detiene si la hay.
  3. Entrena (val sirve para elegir el mejor punto) y guarda el modelo en --output-dir.
  4. Evalúa en val y test: exactitud, precisión/recall por clase y matriz de confusión.
  5. Mide el tiempo de inferencia por imagen y la memoria de GPU.
  6. Solo copia los pesos a --promote-to si la exactitud en TEST alcanza --min-accuracy.

Código de salida: 0 ok; 2 licencia no permitida o fuga de datos; 3 entrenó pero NO se promovió.
"""
import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..data.leakage import IMG_EXTS, find_cross_split_duplicates
from ..evaluate.classification_metrics import classification_report

# Licencias aceptadas (criterio del proyecto: permisivas). Ver docs/LICENCIAS.md.
ALLOWED_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense",
                    "cc-by-4.0", "openrail"}
# CC BY 4.0 permite uso comercial con ATRIBUCIÓN; OpenRAIL tiene restricciones de uso: se acepta solo si
# el grupo lo revisó. Sirven para modelos/datasets cuya ficha lo declare.
NEEDS_REVIEW = {"cc-by-4.0": "exige atribución", "openrail": "tiene restricciones de uso: revisar"}


def normalize_license(value):
    return " ".join(str(value).lower().replace("_", "-").split()).replace(" ", "-")


def check_license(kind, value):
    """Devuelve la licencia normalizada o lanza ValueError si no es permisiva."""
    norm = normalize_license(value)
    if norm not in ALLOWED_LICENSES:
        raise ValueError(
            f"Licencia {kind} no permitida: {value!r}. Se aceptan: {sorted(ALLOWED_LICENSES)}. "
            "Si la ficha dice otra cosa (no comercial, GPL/AGPL, 'research only'), no se puede usar "
            "para un producto que se comercialice. Ver docs/LICENCIAS.md.")
    return norm


def list_classes(split_dir):
    return sorted(d.name for d in Path(split_dir).iterdir() if d.is_dir())


def list_items(split_dir, classes):
    index = {c: i for i, c in enumerate(classes)}
    items = []
    for cls in list_classes(split_dir):
        if cls not in index:
            raise ValueError(f"La clase {cls!r} de {split_dir} no existe en train: {classes}")
        for f in sorted((Path(split_dir) / cls).iterdir()):
            if f.suffix.lower() in IMG_EXTS and f.is_file():
                items.append((f, index[cls]))
    return items


class FolderClassificationDataset:
    """<raíz>/<clase>/<imagen>. Lee con imdecode (funciona con rutas con tildes en Windows)."""

    def __init__(self, split_dir, classes, processor, augment=False):
        self.items = list_items(split_dir, classes)
        if not self.items:
            raise ValueError(f"{split_dir} no tiene imágenes")
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment:
            rgb = augment_image(rgb)
        pixel_values = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "labels": label}


def augment_image(rgb):
    """Aumentos suaves. NO se espeja (el espejo invierte las letras de las insignias) y el recorte
    aleatorio imita lo que pasa en producción: el vehículo suele salir cortado por el borde del cuadro."""
    h, w = rgb.shape[:2]
    ch, cw = int(h * random.uniform(0.85, 1.0)), int(w * random.uniform(0.85, 1.0))
    y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
    out = rgb[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    out = out * random.uniform(0.8, 1.2) + random.uniform(-20, 20)
    return np.clip(out, 0, 255).astype(np.uint8)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def measure_inference(model, processor, device, images, repeats=20):
    """Tiempo por imagen (una a una, como en producción) y pico de memoria de GPU."""
    import torch
    model.eval()
    sample = [images[i % len(images)] for i in range(repeats)]
    times = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for k, rgb in enumerate(sample + sample[:3]):
            x = processor(images=rgb, return_tensors="pt").to(device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if k >= 3:                       # las primeras pasadas son de calentamiento
                times.append((time.perf_counter() - t0) * 1000.0)
    out = {"device": device, "images": len(times), "mean_ms": statistics.mean(times),
           "median_ms": statistics.median(times), "max_ms": max(times)}
    if device.startswith("cuda"):
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--base-model", required=True,
                   help="Modelo base (id de Hugging Face o carpeta local) con licencia permisiva")
    p.add_argument("--base-model-license", required=True, help="Licencia que declara la ficha del modelo base")
    p.add_argument("--dataset-license", required=True, help="Licencia que declara la fuente del dataset")
    p.add_argument("--dataset-source", default="", help="URL o descripción del origen del dataset (se registra)")
    p.add_argument("--output-dir", default=os.path.join("outputs", "clasificador_candidato"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda o cpu para medir la inferencia (por defecto, el disponible)")
    p.add_argument("--leakage-threshold", type=int, default=2,
                   help="Distancia máxima de hash (0-64) para considerar 'casi idéntica' a una imagen. 2 por defecto")
    p.add_argument("--allow-leakage", action="store_true", help="Sigue aunque haya fuga (NO recomendado)")
    p.add_argument("--min-accuracy", type=float, default=None,
                   help="Exactitud mínima en TEST para promover los pesos (obligatorio con --promote-to)")
    p.add_argument("--promote-to", default=None,
                   help="Carpeta donde dejar el modelo SOLO si cumple --min-accuracy (p. ej. weights/vehicle_classifier)")
    args = p.parse_args(argv)

    # 1) licencias (antes de cargar nada pesado)
    try:
        base_license = check_license("del modelo base", args.base_model_license)
        data_license = check_license("del dataset", args.dataset_license)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for kind, lic in (("modelo base", base_license), ("dataset", data_license)):
        if lic in NEEDS_REVIEW:
            print(f"AVISO: la licencia del {kind} ({lic}) {NEEDS_REVIEW[lic]}.", file=sys.stderr)
    if args.promote_to and args.min_accuracy is None:
        p.error("--promote-to requiere --min-accuracy: no se promueven pesos sin un criterio de aceptación")

    data = Path(args.data_dir)
    train_dir, val_dir, test_dir = data / "train", data / "val", data / "test"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            print(f"ERROR: falta {d}", file=sys.stderr)
            return 2
    split_dirs = {"train": train_dir, "val": val_dir}
    if test_dir.is_dir():
        split_dirs["test"] = test_dir
    classes = list_classes(train_dir)
    if len(classes) < 2:
        print("ERROR: hacen falta al menos 2 clases en train", file=sys.stderr)
        return 2

    # 2) fuga de datos entre particiones
    dups = find_cross_split_duplicates(split_dirs, args.leakage_threshold)
    if dups:
        print(f"ERROR: FUGA DE DATOS: {len(dups)} par(es) de imágenes repetidas entre particiones. Ejemplos:",
              file=sys.stderr)
        for d in dups[:10]:
            print(f"  [{d['tipo']}, distancia {d['distancia']}] {d['split_a']}: {d['a']}  <->  "
                  f"{d['split_b']}: {d['b']}", file=sys.stderr)
        if not args.allow_leakage:
            print("Repartí por vehículo con `python -m alpr.data.splits classes` o quitá los duplicados.",
                  file=sys.stderr)
            return 2

    import torch
    from transformers import (AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments,
                              set_seed)
    import inspect

    set_seed(args.seed)
    random.seed(args.seed)
    id2label = dict(enumerate(classes))
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model, num_labels=len(classes), id2label=id2label,
        label2id={c: i for i, c in id2label.items()}, ignore_mismatched_sizes=True)

    train_ds = FolderClassificationDataset(train_dir, classes, processor, augment=args.augment)
    val_ds = FolderClassificationDataset(val_dir, classes, processor)
    test_ds = FolderClassificationDataset(test_dir, classes, processor) if "test" in split_dirs else None

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}

    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    warmup = {"warmup_ratio": 0.1} if "warmup_ratio" in sig else {"warmup_steps": 0.1}
    out_dir = Path(args.output_dir)
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay, lr_scheduler_type="cosine",
        fp16=bool(args.fp16 and torch.cuda.is_available()), dataloader_num_workers=args.workers,
        save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="accuracy", greater_is_better=True, logging_steps=20,
        remove_unused_columns=False, report_to="none", seed=args.seed, **warmup, **{eval_key: "epoch"})
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    trainer.train()

    # 3) guardar el mejor modelo (con el formato que lee alpr.classify)
    final_dir = out_dir / "modelo"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # 4) evaluación por partición
    def evaluate(ds):
        pred = trainer.predict(ds)
        y_pred = np.argmax(pred.predictions, axis=-1).tolist()
        return classification_report([label for _, label in ds.items], y_pred, classes)

    metrics = {"val": evaluate(val_ds)}
    if test_ds is not None:
        metrics["test"] = evaluate(test_ds)

    # 5) tiempo de inferencia y memoria
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)
    samples = []
    for path, _ in (test_ds or val_ds).items[:20]:
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        samples.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    metrics["inference"] = measure_inference(trainer.model, processor, device, samples)

    reference = "test" if "test" in metrics else "val"
    accuracy = metrics[reference]["accuracy"]
    promoted = bool(args.promote_to and test_ds is not None and accuracy >= args.min_accuracy)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "confusion_matrix_test.csv" if test_ds is not None else out_dir / "confusion_matrix_val.csv",
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["real\\predicha"] + classes)
        for name, row in zip(classes, metrics[reference]["confusion_matrix"]):
            w.writerow([name] + row)
    provenance = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "modelo_base": args.base_model, "licencia_modelo_base": base_license,
        "dataset_origen": args.dataset_source, "licencia_dataset": data_license,
        "clases": classes, "imagenes": {"train": len(train_ds), "val": len(val_ds),
                                        "test": len(test_ds) if test_ds is not None else 0},
        "semilla": args.seed, "versiones": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "criterio": {"particion_de_referencia": reference, "exactitud": accuracy,
                     "minimo_exigido": args.min_accuracy, "promovido": promoted},
        "argumentos": {k: v for k, v in vars(args).items()},
    }
    try:
        import transformers
        provenance["versiones"]["transformers"] = transformers.__version__
    except Exception:
        pass
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False, default=str)

    print(f"Exactitud en {reference}: {accuracy:.3f} | clases: {len(classes)} | "
          f"inferencia: {metrics['inference']['median_ms']:.1f} ms/imagen en {device}")
    for name, v in metrics[reference]["per_class"].items():
        rec = "-" if v["recall"] is None else f"{v['recall']:.2f}"
        print(f"  {name:<30} recall {rec}  ({v['support']} imágenes)")

    # 6) promoción solo si cumple el criterio de aceptación
    if args.promote_to:
        if promoted:
            dest = Path(args.promote_to)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(final_dir, dest, dirs_exist_ok=True)
            shutil.copy2(out_dir / "train_config.json", dest / "train_config.json")
            print(f"PROMOVIDO a {dest} (exactitud {accuracy:.3f} >= {args.min_accuracy})")
        else:
            why = "no hay carpeta test" if test_ds is None else f"exactitud {accuracy:.3f} < {args.min_accuracy}"
            print(f"NO PROMOVIDO: {why}. Los pesos quedaron solo en {final_dir}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
