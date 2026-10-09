"""ENTRENAMIENTO del detector de patentes (RT-DETR, una clase: "plate").

Es lo único que se entrena en todo el sistema: RF-DETR (vehículos) y EasyOCR
vienen preentrenados. Fine-tuning con la API de Hugging Face:

    python -m alpr.train.train_rtdetr \
        --train-ann data/detector/train.json --train-images data/CCPD2019 \
        --val-ann   data/detector/val.json   --val-images   data/CCPD2019 \
        --output-dir outputs/rtdetr-ccpd

Deja el modelo en <output-dir>/final_model (la ruta que espera el pipeline) y
el log de la corrida en <output-dir>/train_config.json.

Solo se usan train y val. La partición `test` no se toca acá: se evalúa una
sola vez con `python -m alpr.evaluate.detector_eval`. La partición val sirve
para elegir el mejor checkpoint (eval_loss) y para detener el sobreajuste.
"""
import argparse
import inspect
import json
import os
import random

import cv2
import numpy as np


class CocoPlateDataset:
    """Dataset COCO de una clase. Devuelve lo que espera RTDetrForObjectDetection:
    `pixel_values` + `labels` (class_labels y boxes cxcywh normalizadas, que
    genera el RTDetrImageProcessor)."""

    def __init__(self, ann_path, images_dir, processor, augment=False):
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.by_image = {}
        for a in coco["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)
        self.images_dir = images_dir
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = cv2.imread(os.path.join(self.images_dir, info["file_name"]))
        if img is None:
            raise FileNotFoundError(os.path.join(self.images_dir, info["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        anns = [dict(bbox=list(a["bbox"]), category_id=int(a["category_id"]),
                     area=float(a.get("area", a["bbox"][2] * a["bbox"][3])), iscrowd=0)
                for a in self.by_image.get(info["id"], [])]

        if self.augment:
            img = _photometric_jitter(img)

        enc = self.processor(
            images=img,
            annotations={"image_id": info["id"], "annotations": anns},
            return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def _photometric_jitter(img):
    """Aumento fotométrico suave (brillo/contraste). Se evita el flip horizontal:
    espejaría el texto de la chapa y no representa nada real."""
    alpha = random.uniform(0.75, 1.25)   # contraste
    beta = random.uniform(-25, 25)       # brillo
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def collate_fn(batch):
    import torch
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": [b["labels"] for b in batch]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--train-images", required=True)
    p.add_argument("--val-ann", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--output-dir", default=os.path.join("outputs", "rtdetr-ccpd"))
    p.add_argument("--base-model", default="PekingU/rtdetr_r50vd",
                   help="Checkpoint preentrenado de partida (id de HF o ruta local)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-images", type=int, default=0,
                   help="Para pruebas rápidas: 0 = todas")
    args = p.parse_args(argv)

    import torch
    from transformers import (RTDetrForObjectDetection, RTDetrImageProcessor,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    random.seed(args.seed)

    processor = RTDetrImageProcessor.from_pretrained(args.base_model)
    id2label = {0: "plate"}
    model = RTDetrForObjectDetection.from_pretrained(
        args.base_model, id2label=id2label, label2id={"plate": 0},
        num_labels=1, ignore_mismatched_sizes=True)

    train_ds = CocoPlateDataset(args.train_ann, args.train_images, processor, augment=args.augment)
    val_ds = CocoPlateDataset(args.val_ann, args.val_images, processor, augment=False)
    if args.max_train_images:
        train_ds.images = train_ds.images[:args.max_train_images]

    # transformers renombró evaluation_strategy -> eval_strategy.
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    # transformers 5 quitó warmup_ratio: warmup_steps acepta un float en [0, 1) como proporción.
    warmup = {"warmup_ratio": 0.05} if "warmup_ratio" in sig else {"warmup_steps": 0.05}

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    targs = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        dataloader_num_workers=args.workers,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="none",
        seed=args.seed,
        **warmup,
        **{eval_key: "epoch"},
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate_fn)
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "n_train": len(train_ds), "n_val": len(val_ds),
                   "log_history": trainer.state.log_history}, f, indent=2, default=str)
    print(f"Modelo final -> {final_dir}")


if __name__ == "__main__":
    main()
