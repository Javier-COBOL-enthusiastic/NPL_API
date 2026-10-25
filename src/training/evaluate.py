#!/usr/bin/env python
"""Evaluación y calibración del umbral de no-respuesta (Fase 5 del plan).

Qué hace:
  1. Carga final_model/ desde Drive y recorre la validación de SQuAD 2.0.
  2. Por cada ejemplo recorre TODAS sus ventanas (stride) y calcula:
        span_score = mejor start_logit + end_logit dentro del contexto
        null_score = start_logit[CLS] + end_logit[CLS]   (mínimo entre ventanas)
        delta      = null_score - span_score
     Se predice null cuando  delta > umbral.
  3. EM / F1 oficiales de SQuAD 2.0 (global, HasAns y NoAns).
  4. Barrido de umbrales sobre validación -> elige el que maximiza F1.
  5. Comprueba que cada respuesta no nula es un fragmento exacto del contexto.
  6. Corre public_sample.json con el umbral elegido.

Salidas (en Drive):
  results/validation_predictions.json   results/threshold_sweep.csv
  results/eval_summary.json             results/public_sample_results.json
  results/postprocess_config.json       (copiado también a final_model/)

Las funciones run_inference() y apply_threshold() están pensadas para
reutilizarse tal cual en service/inference.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import shutil
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DATASET_ID = "rajpurkar/squad_v2"
NO_SPAN = -1e9  # score centinela cuando una ventana no tiene ningún span válido

log = logging.getLogger("evaluate")


# --------------------------------------------------------------------------- #
# Inferencia por ventanas -> candidato por ejemplo
# --------------------------------------------------------------------------- #
def _infer_chunk(model, tokenizer, questions, contexts, max_len, stride,
                 batch_size, device, n_best, max_answer_len):
    enc = tokenizer(
        [q.lstrip() for q in questions],
        contexts,
        truncation="only_second",
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = enc["overflow_to_sample_mapping"]
    input_keys = [k for k in tokenizer.model_input_names if k in enc]
    n_feat = len(enc["input_ids"])
    start_logits, end_logits = [None] * n_feat, [None] * n_feat

    for b in range(0, n_feat, batch_size):
        batch = {k: torch.tensor(enc[k][b:b + batch_size], device=device) for k in input_keys}
        # Recorta el padding sobrante del batch (acelera mucho en CPU).
        longest = int(batch["attention_mask"].sum(1).max())
        batch = {k: v[:, :longest] for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            out = model(**batch)
        s = out.start_logits.float().cpu().numpy()
        e = out.end_logits.float().cpu().numpy()
        for j in range(s.shape[0]):
            start_logits[b + j], end_logits[b + j] = s[j], e[j]

    windows_of = [[] for _ in questions]
    for feat_idx, ex_idx in enumerate(sample_map):
        windows_of[ex_idx].append(feat_idx)

    results = []
    for ex_idx, feats in enumerate(windows_of):
        ctx = contexts[ex_idx]
        null_score = None
        best_score, best_text, best_start = NO_SPAN, "", None
        for f in feats:
            s, e = start_logits[f], end_logits[f]
            offsets = enc["offset_mapping"][f]
            seq_ids = enc.sequence_ids(f)

            ns = float(s[0] + e[0])  # [CLS] está siempre en la posición 0
            null_score = ns if null_score is None else min(null_score, ns)

            for si in np.argsort(s)[::-1][:n_best]:
                if seq_ids[si] != 1:
                    continue
                for ei in np.argsort(e)[::-1][:n_best]:
                    if seq_ids[ei] != 1 or ei < si or ei - si + 1 > max_answer_len:
                        continue
                    sc = float(s[si] + e[ei])
                    if sc > best_score:
                        best_score = sc
                        best_start = offsets[si][0]
                        best_text = ctx[offsets[si][0]:offsets[ei][1]]
        results.append({
            "answer": best_text,
            "start": best_start,
            "span_score": best_score,
            "null_score": null_score,
            "delta": null_score - best_score,
            "n_windows": len(feats),
        })
    return results


def run_inference(model, tokenizer, questions, contexts, max_len=384, stride=128,
                  batch_size=64, device=None, n_best=20, max_answer_len=30,
                  chunk_size=256, show_progress=False):
    """Devuelve, por ejemplo, el mejor span y su delta (sin aplicar umbral)."""
    device = device or next(model.parameters()).device
    results = []
    starts = range(0, len(questions), chunk_size)
    for c in tqdm(starts, disable=not show_progress, desc="Inferencia"):
        results.extend(_infer_chunk(
            model, tokenizer, questions[c:c + chunk_size], contexts[c:c + chunk_size],
            max_len, stride, batch_size, device, n_best, max_answer_len))
    return results


def decision_score(delta: float, threshold: float) -> float:
    """Confianza en [0, 1] de la decisión tomada (respuesta o null).

    Crece con la distancia entre delta y el umbral: 0.5 = en la frontera.
    """
    return 1.0 / (1.0 + math.exp(-min(abs(delta - threshold), 50.0)))


def apply_threshold(pred: dict, threshold: float):
    """-> (answer | None, start | None, score). Contrato de la API."""
    score = decision_score(pred["delta"], threshold)
    if pred["delta"] > threshold or pred["start"] is None or not pred["answer"]:
        return None, None, score
    return pred["answer"], int(pred["start"]), score


# --------------------------------------------------------------------------- #
# Métricas oficiales de SQuAD 2.0
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    g, p = _tokens(gold), _tokens(pred)
    if not g or not p:
        return float(g == p)
    same = sum((Counter(g) & Counter(p)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_answers_for(answers: dict) -> list[str]:
    golds = [t for t in answers["text"] if normalize_answer(t)]
    return golds or [""]


def score_against(pred_text: str, golds: list[str]) -> tuple[float, float]:
    return (max(compute_exact(g, pred_text) for g in golds),
            max(compute_f1(g, pred_text) for g in golds))


# --------------------------------------------------------------------------- #
# Barrido de umbral
# --------------------------------------------------------------------------- #
def _mean(x: np.ndarray) -> float:
    return float(x.mean() * 100) if x.size else float("nan")


def threshold_sweep(deltas, span_em, span_f1, null_em, null_f1, has_ans) -> list[dict]:
    grid = np.unique(np.concatenate([
        np.quantile(deltas, np.linspace(0, 1, 501)),
        [0.0, deltas.min() - 1.0, deltas.max() + 1.0],
    ]))
    rows = []
    for t in grid:
        use_null = deltas > t
        em = np.where(use_null, null_em, span_em)
        f1 = np.where(use_null, null_f1, span_f1)
        rows.append({
            "threshold": float(t),
            "exact": _mean(em),
            "f1": _mean(f1),
            "has_ans_f1": _mean(f1[has_ans]),
            "no_ans_acc": _mean(f1[~has_ans]),
            "null_rate": _mean(use_null.astype(float)),
        })
    return rows


def metrics_at(rows: list[dict], threshold: float) -> dict:
    return min(rows, key=lambda r: abs(r["threshold"] - threshold))


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def count_span_violations(contexts, preds, threshold) -> int:
    bad = 0
    for ctx, p in zip(contexts, preds):
        ans, start, _ = apply_threshold(p, threshold)
        if ans is not None and ctx[start:start + len(ans)] != ans:
            bad += 1
    return bad


def load_public_sample(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if not lists:
            raise ValueError("public_sample.json: no encontré una lista de casos.")
        data = lists[0]
    return [{"id": str(it.get("id", i)), "context": it["context"], "question": it["question"]}
            for i, it in enumerate(data)]


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_run_config(root: Path) -> dict:
    p = root / "config" / "run_config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Evaluación y calibración de null (SQuAD 2.0)")
    p.add_argument("--drive_root", default="/content/drive/MyDrive/ExtractiveQA")
    p.add_argument("--model_dir", default=None, help="Por defecto: <drive_root>/final_model")
    p.add_argument("--public_sample", default=None, help="Ruta a public_sample.json")
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--doc_stride", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_len", type=int, default=30)
    p.add_argument("--max_examples", type=int, default=0,
                   help="0 = toda la validación. Útil solo para pruebas rápidas.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = parse_args()
    root = Path(args.drive_root)
    model_dir = Path(args.model_dir) if args.model_dir else root / "final_model"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    hp = load_run_config(root).get("hyperparameters", {})
    max_len = args.max_seq_length or hp.get("max_seq_length", 384)
    stride = args.doc_stride or hp.get("doc_stride", 128)
    log.info("max_seq_length=%s doc_stride=%s", max_len, stride)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device).eval()

    from datasets import load_dataset  # import tardío: facilita probar el resto
    val = load_dataset(DATASET_ID, split="validation")
    is_subset = bool(args.max_examples) and args.max_examples < len(val)
    if is_subset:
        val = val.select(range(args.max_examples))
        log.warning("Usando solo %d ejemplos: NO uses este umbral como definitivo.", len(val))

    ids, questions, contexts = list(val["id"]), list(val["question"]), list(val["context"])
    golds = [gold_answers_for(a) for a in val["answers"]]
    has_ans = np.array([golds[i] != [""] for i in range(len(val))])

    t0 = time.time()
    preds = run_inference(model, tokenizer, questions, contexts, max_len, stride,
                          args.batch_size, device, args.n_best, args.max_answer_len,
                          show_progress=True)
    log.info("Inferencia: %.1f s (%.1f ms/ejemplo)", time.time() - t0,
             (time.time() - t0) / len(val) * 1000)

    # Puntajes por ejemplo: si se predice el span, y si se predice null ("").
    span = [score_against(p["answer"], g) for p, g in zip(preds, golds)]
    null = [score_against("", g) for g in golds]
    span_em, span_f1 = np.array([s[0] for s in span]), np.array([s[1] for s in span])
    null_em, null_f1 = np.array([s[0] for s in null]), np.array([s[1] for s in null])
    deltas = np.array([p["delta"] for p in preds])

    rows = threshold_sweep(deltas, span_em, span_f1, null_em, null_f1, has_ans)
    best = max(rows, key=lambda r: r["f1"])
    at_zero = metrics_at(rows, 0.0)
    threshold = best["threshold"]
    log.info("delta=0   -> EM %.2f | F1 %.2f | HasAns F1 %.2f | NoAns acc %.2f",
             at_zero["exact"], at_zero["f1"], at_zero["has_ans_f1"], at_zero["no_ans_acc"])
    log.info("MEJOR t=%.3f -> EM %.2f | F1 %.2f | HasAns F1 %.2f | NoAns acc %.2f",
             threshold, best["exact"], best["f1"], best["has_ans_f1"], best["no_ans_acc"])

    violations = count_span_violations(contexts, preds, threshold)
    log.info("Respuestas que NO son fragmento exacto del contexto: %d", violations)

    # Guardado de resultados de validación.
    write_csv(results_dir / "threshold_sweep.csv", rows)
    write_json(results_dir / "validation_predictions.json", [
        {"id": i, "has_answer": bool(h), "answer": p["answer"], "start": p["start"],
         "span_score": p["span_score"], "null_score": p["null_score"],
         "delta": p["delta"], "n_windows": p["n_windows"]}
        for i, h, p in zip(ids, has_ans, preds)
    ])
    write_json(results_dir / "eval_summary.json", {
        "examples": len(val), "subset": is_subset,
        "at_delta_0": at_zero, "at_best_threshold": best,
        "span_violations": violations,
        "answerable": int(has_ans.sum()), "unanswerable": int((~has_ans).sum()),
    })

    post_cfg = {
        "threshold": threshold,
        "rule": "predict null if delta > threshold, delta = null_score - span_score",
        "score": "sigmoid(|delta - threshold|)",
        "selection_metric": "SQuAD 2.0 F1 on validation",
        "max_seq_length": max_len, "doc_stride": stride,
        "n_best": args.n_best, "max_answer_len": args.max_answer_len,
        "val_f1": best["f1"], "val_exact": best["exact"],
        "validation_examples": len(val),
    }
    write_json(results_dir / "postprocess_config.json", post_cfg)
    if not is_subset:  # el umbral viaja junto con el modelo
        shutil.copy(results_dir / "postprocess_config.json", model_dir / "postprocess_config.json")

    # Muestra pública.
    if args.public_sample:
        items = load_public_sample(args.public_sample)
        out = run_inference(model, tokenizer, [i["question"] for i in items],
                            [i["context"] for i in items], max_len, stride,
                            args.batch_size, device, args.n_best, args.max_answer_len)
        public = []
        for it, p in zip(items, out):
            ans, start, score = apply_threshold(p, threshold)
            public.append({"id": it["id"], "answer": ans, "start": start, "score": score})
            n_tok = len(tokenizer(it["context"])["input_ids"])
            log.info("[%s] answer=%r start=%s score=%.3f | tokens_ctx=%d ventanas=%d delta=%.2f",
                     it["id"], ans, start, score, n_tok, p["n_windows"], p["delta"])
        write_json(results_dir / "public_sample_results.json", public)
    log.info("Resultados guardados en %s", results_dir)


if __name__ == "__main__":
    main()
