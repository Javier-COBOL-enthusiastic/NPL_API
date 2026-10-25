#!/usr/bin/env python
"""Fine-tuning de un encoder para Extractive QA (SQuAD 2.0) en Colab + Drive.

Cubre las fases 3 y 4 del plan:
  - Fase 3: rutas en Drive, run_config.json, logs.
  - Fase 4: preprocesado con offsets/stride, ejemplos sin respuesta -> [CLS],
            checkpoints reanudables (modelo + optimizador + scheduler),
            métricas CSV en Drive y exportación a final_model/.

La evaluación completa (EM/F1 y umbral de null) va en evaluate.py (fase 5).

Uso desde el notebook:
    from google.colab import drive
    drive.mount("/content/drive")
    !python train.py --train_samples 25000 --epochs 2          # piloto
    !python train.py --epochs 3                                 # corrida completa

Si la sesión se corta, vuelve a montar Drive y ejecuta EXACTAMENTE el mismo
comando: el script detecta el último checkpoint válido y reanuda.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ExtractiveQA"
DEFAULT_MODEL = "distilbert/distilbert-base-uncased"
DATASET_ID = "rajpurkar/squad_v2"

log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Argumentos
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning Extractive QA (SQuAD 2.0)")
    p.add_argument("--drive_root", default=DEFAULT_DRIVE_ROOT)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_samples", type=int, default=0,
                   help="0 = todo el train. Usa 20000-30000 para el piloto.")
    p.add_argument("--eval_samples", type=int, default=2000,
                   help="Subconjunto de validación para eval_loss durante el "
                        "entrenamiento (0 = todo). El EM/F1 completo va en evaluate.py.")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--no_resume", action="store_true",
                   help="Ignora checkpoints existentes y empieza de cero.")
    p.add_argument("--allow_squad_checkpoint", action="store_true",
                   help="Desactiva la guarda contra checkpoints ya fine-tuned en SQuAD.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Fase 3: rutas, logging y configuración de la corrida
# --------------------------------------------------------------------------- #
def setup_paths(drive_root: str) -> dict[str, Path]:
    root = Path(drive_root)
    if str(root).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        raise RuntimeError(
            "Drive no está montado. En el notebook ejecuta primero:\n"
            "    from google.colab import drive; drive.mount('/content/drive')"
        )
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "final_model": root / "final_model",
        "logs": root / "logs",
        "results": root / "results",
        "config": root / "config",
    }
    for key, d in paths.items():
        if key != "root":
            d.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_file: Path) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def guard_base_checkpoint(model_name: str, allow: bool) -> None:
    """El plan prohíbe partir de un checkpoint ya fine-tuned en SQuAD."""
    if "squad" in model_name.lower() and not allow:
        raise ValueError(
            f"'{model_name}' parece un checkpoint fine-tuned en SQuAD, "
            "lo cual no está permitido como punto de partida."
        )


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Fase 4: preprocesado
# --------------------------------------------------------------------------- #
def prepare_train_features(examples, tokenizer, max_len: int, stride: int):
    """Tokeniza (pregunta, contexto) con ventanas deslizantes y calcula
    start_positions / end_positions.

    - Pregunta sin respuesta           -> ambos apuntan a [CLS] (índice 0).
    - Ventana que no contiene la respuesta -> también [CLS].
    """
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        answers = examples["answers"][sample_map[i]]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Rango de tokens que pertenecen al contexto (sequence_id == 1).
        token_start = 0
        while sequence_ids[token_start] != 1:
            token_start += 1
        token_end = len(input_ids) - 1
        while sequence_ids[token_end] != 1:
            token_end -= 1

        # ¿La respuesta cae completa dentro de esta ventana?
        if not (offsets[token_start][0] <= start_char and offsets[token_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)
            while offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def build_datasets(args, tokenizer):
    raw = load_dataset(DATASET_ID)
    train_raw, val_raw = raw["train"], raw["validation"]

    if args.train_samples and args.train_samples < len(train_raw):
        train_raw = train_raw.shuffle(seed=args.seed).select(range(args.train_samples))
    if args.eval_samples and args.eval_samples < len(val_raw):
        val_raw = val_raw.shuffle(seed=args.seed).select(range(args.eval_samples))

    def prep(batch):
        return prepare_train_features(batch, tokenizer, args.max_seq_length, args.doc_stride)

    train_ds = train_raw.map(prep, batched=True, remove_columns=train_raw.column_names,
                             desc="Tokenizando train")
    val_ds = val_raw.map(prep, batched=True, remove_columns=val_raw.column_names,
                         desc="Tokenizando validación")
    counts = {
        "train_examples": len(train_raw),
        "train_features": len(train_ds),
        "eval_examples": len(val_raw),
        "eval_features": len(val_ds),
    }
    return train_ds, val_ds, counts


# --------------------------------------------------------------------------- #
# Fase 4: métricas CSV en Drive y reanudación
# --------------------------------------------------------------------------- #
class DriveCsvLogger(TrainerCallback):
    """Añade cada log del Trainer a CSVs en Drive (modo append: sobrevive a
    desconexiones y a reanudaciones)."""

    TRAIN_FIELDS = ["step", "epoch", "loss", "learning_rate", "grad_norm"]
    EVAL_FIELDS = ["step", "epoch", "eval_loss", "eval_runtime"]

    def __init__(self, log_dir: Path):
        self.train_csv = log_dir / "train_metrics.csv"
        self.eval_csv = log_dir / "eval_metrics.csv"

    @staticmethod
    def _append(path: Path, fields: list[str], row: dict) -> None:
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": state.epoch, **logs}
        if "eval_loss" in logs:
            self._append(self.eval_csv, self.EVAL_FIELDS, row)
        elif "loss" in logs:
            self._append(self.train_csv, self.TRAIN_FIELDS, row)


def find_resumable_checkpoint(ckpt_dir: Path) -> str | None:
    """Último checkpoint COMPLETO. Una desconexión a mitad de escritura puede
    dejar una carpeta incompleta; esas se saltan."""
    candidates = []
    for d in ckpt_dir.glob("checkpoint-*"):
        try:
            candidates.append((int(d.name.split("-")[1]), d))
        except (IndexError, ValueError):
            continue
    for _, d in sorted(candidates, reverse=True):
        has_weights = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
        needed = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
        if has_weights and all((d / n).exists() for n in needed):
            return str(d)
        log.warning("Checkpoint incompleto, se ignora: %s", d)
    return None


def build_training_args(args, paths, num_train_features: int) -> TrainingArguments:
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    # `warmup_ratio` se eliminó en Transformers recientes: se calculan los pasos
    # a mano, lo que funciona igual en todas las versiones.
    steps_per_epoch = math.ceil(num_train_features / (args.batch_size * args.grad_accum))
    warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)
    log.info("Pasos/época: %d | warmup_steps: %d", steps_per_epoch, warmup_steps)
    kwargs = dict(
        output_dir=str(paths["checkpoints"]),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fp16=use_cuda and not bf16,
        bf16=bf16,
        # Evaluación y guardado alineados por pasos (requisito de
        # load_best_model_at_end). eval_loss se calcula sobre un subconjunto.
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to="none",
        dataloader_num_workers=2,
    )
    # El nombre cambió entre versiones de Transformers.
    eval_key = ("eval_strategy"
                if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
                else "evaluation_strategy")
    kwargs[eval_key] = "steps"

    # Descarta (con aviso) argumentos que esta versión de Transformers ya no tenga.
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    for k in [k for k in kwargs if k not in valid]:
        log.warning("TrainingArguments ya no acepta '%s'; se omite.", k)
        kwargs.pop(k)
    return TrainingArguments(**kwargs)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    paths = setup_paths(args.drive_root)
    setup_logging(paths["logs"] / "training_log.txt")
    guard_base_checkpoint(args.model_name, args.allow_squad_checkpoint)
    set_seed(args.seed)

    env = environment_info()
    log.info("Entorno: %s", env)
    if not env["cuda_available"]:
        log.warning("No hay GPU. El entrenamiento en CPU será extremadamente lento.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Se necesita un tokenizer rápido para return_offsets_mapping.")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Modelo: %s | parámetros: %s", args.model_name, f"{n_params:,}")

    train_ds, val_ds, counts = build_datasets(args, tokenizer)
    log.info("Datos: %s", counts)

    write_json(paths["config"] / "run_config.json", {
        "base_checkpoint": args.model_name,
        "dataset": DATASET_ID,
        "num_parameters": n_params,
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "drive_root"},
        "data_counts": counts,
        "environment": env,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # `tokenizer` fue renombrado a `processing_class` en Transformers recientes.
    tok_key = ("processing_class"
               if "processing_class" in inspect.signature(Trainer.__init__).parameters
               else "tokenizer")
    trainer = Trainer(
        model=model,
        args=build_training_args(args, paths, len(train_ds)),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        callbacks=[DriveCsvLogger(paths["logs"])],
        **{tok_key: tokenizer},
    )

    resume = None if args.no_resume else find_resumable_checkpoint(paths["checkpoints"])
    if resume:
        log.info("Reanudando desde %s", resume)
    else:
        log.info("Entrenamiento desde cero.")

    train_result = trainer.train(resume_from_checkpoint=resume)

    # Con load_best_model_at_end=True, el trainer ya tiene el mejor checkpoint.
    trainer.save_model(str(paths["final_model"]))
    tokenizer.save_pretrained(str(paths["final_model"]))

    final_eval = trainer.evaluate()
    write_json(paths["logs"] / "final_metrics.json", {
        "train": train_result.metrics,
        "eval": final_eval,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    write_json(paths["logs"] / "log_history.json", {"log_history": trainer.state.log_history})
    log.info("Listo. Mejor checkpoint: %s | modelo final en %s",
             trainer.state.best_model_checkpoint, paths["final_model"])


if __name__ == "__main__":
    main()
