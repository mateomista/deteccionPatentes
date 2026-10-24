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
import re

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

        anns = [dict(bbox=list(a["bbox"]), category_id=0,  # una sola clase ("plate"); se ignora
                     # el category_id real del COCO json (Roboflow suele numerar la clase real
                     # desde 1, dejando 0 para un placeholder de supercategoria "none")
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


def _remap_legacy_rtdetr_state_dict(state_dict):
    """Traduce nombres de peso de un checkpoint de RT-DETR guardado con una
    version mas vieja de `transformers` (donde el submodulo de auto-atencion
    del encoder se llamaba "encoder.encoder" en vez de "encoder.aifi", y las
    capas usaban "fc1"/"fc2" + "self_attn.out_proj" sueltos en vez de
    "mlp.fc1"/"mlp.fc2" + "self_attn.o_proj") a los nombres que usa la version
    instalada. Sin esto, from_pretrained() deja la carga "a medias": carga el
    resto del modelo bien pero reinicializa al azar la auto-atencion del
    encoder y las capas FFN/proyeccion de salida del decoder, sin mas aviso
    que un log de "missing/unexpected keys" facil de pasar por alto -- y eso
    tira a la basura buena parte de un fine-tuning anterior."""
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k.replace("model.encoder.encoder.", "model.encoder.aifi.")
        nk = nk.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        nk = re.sub(r"(\.layers\.\d+)\.fc1\.", r"\1.mlp.fc1.", nk)
        nk = re.sub(r"(\.layers\.\d+)\.fc2\.", r"\1.mlp.fc2.", nk)
        new_state_dict[nk] = v
    return new_state_dict


def _load_legacy_checkpoint_state_dict(checkpoint_dir):
    import torch
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        sd = load_file(st_path)
    else:
        sd = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu")
    return _remap_legacy_rtdetr_state_dict(sd)


def _load_base_model(base_model, model_cls):
    """Carga el modelo base evitando una reinicializacion innecesaria de la
    cabeza de clasificacion, y remapeando nombres de clave legacy cuando el
    checkpoint local viene de una version mas vieja de `transformers`."""
    cfg_path = os.path.join(base_model, "config.json")
    already_one_class = False
    if os.path.isdir(base_model) and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
        already_one_class = len(base_cfg.get("id2label", {})) == 1

    if already_one_class:
        model = model_cls.from_pretrained(base_model)
        legacy_sd = _load_legacy_checkpoint_state_dict(base_model)
        missing, unexpected = model.load_state_dict(legacy_sd, strict=False)
        if missing or unexpected:
            print(f"[WARN] tras remapear claves legacy, quedan sin cargar: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected. "
                  f"Primeras missing: {missing[:5]}", flush=True)
        else:
            print("[INFO] checkpoint legacy remapeado y cargado sin faltantes.", flush=True)
        return model
    return model_cls.from_pretrained(
        base_model, id2label={0: "plate"}, label2id={"plate": 0},
        num_labels=1, ignore_mismatched_sizes=True)
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

    print("[DIAG] cargando processor...", flush=True)
    processor = RTDetrImageProcessor.from_pretrained(args.base_model)
    print("[DIAG] cargando modelo...", flush=True)
    model = _load_base_model(args.base_model, RTDetrForObjectDetection)

    print("[DIAG] construyendo train_ds...", flush=True)
    train_ds = CocoPlateDataset(args.train_ann, args.train_images, processor, augment=args.augment)
    print("[DIAG] construyendo val_ds...", flush=True)
    val_ds = CocoPlateDataset(args.val_ann, args.val_images, processor, augment=False)
    if args.max_train_images:
        train_ds.images = train_ds.images[:args.max_train_images]
    print(f"[DIAG] datasets listos: train={len(train_ds)} val={len(val_ds)}", flush=True)

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

    print("[DIAG] TrainingArguments listos, creando Trainer...", flush=True)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collate_fn)
    print("[DIAG] Trainer creado, probando __getitem__ sobre 1 sample de train_ds...", flush=True)
    sample = train_ds[0]
    print("[DIAG] sample OK, pixel_values shape:", sample["pixel_values"].shape, flush=True)
    print("[DIAG] arrancando trainer.train()...", flush=True)
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