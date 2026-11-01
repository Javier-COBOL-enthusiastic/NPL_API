# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `Imanes` |
| Members | Dennys Jefferson Molina Abrego; Javier Ernesto Moran Rosales; Diana Raquel Carballo Castillo; Jorge Nicolás Saade Cuestas |
| Repository URL | https://github.com/Javier-COBOL-enthusiastic/NPL_API |
| `submission` commit hash | [COMPLETAR] |
| Public service URL used at the dry run | https://underwent-purge-chase.ngrok-free.dev (HTTPS dry run completed; scope below) |
| AI tools used (which, and for what) | Codex: API, authentication, tests, Windows/ngrok launcher, local verification and report drafting. Claude: training notebook, `train.py` and the evaluation script (`evaluate.py` in this repository). |

## 1. Model choice and justification

**Base checkpoint:** `distilbert/distilbert-base-uncased`, fine-tuned with a question-answering head, not a checkpoint already trained on SQuAD. Its six-layer encoder has 66.36 million parameters and a measured 0.204 s mean local request time. The final validation F1 is 70.39%; this is a quality/size/latency trade-off, not evidence of superiority over untested alternatives.

**Alternatives considered:** a 12-layer base encoder (BERT-base / RoBERTa-base, approximately 110–125M parameters). It was not trained: we estimated roughly twice the CPU cost per window and judged it risky against the 5 s limit before measuring performance. DistilBERT's measured 0.20 s mean later showed substantial headroom; rejection was conservative, and any accuracy gain from a larger model remains untested.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Full `rajpurkar/squad_v2` train: 130,319 examples; 131,754 tokenized windows |
| Validation data (what you validated on, how many examples) | Official SQuAD 2.0 validation split: 11,873 examples (5,928 answerable, 5,945 unanswerable). A fixed random 2,000-example subset of that same split was used for training-time `eval_loss`. Final EM/F1, checkpoint selection and null-threshold calibration all use the full 11,873 examples. There is no separate held-out test set, so these are potentially optimistic selection estimates, not independent test results. |
| Epochs | 3 |
| Learning rate and schedule (warmup?) | `3e-5`; linear schedule (Trainer default); 10% warmup, 4,941 steps |
| Batch size | 8 per device; gradient accumulation 1 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | Google Colab, NVIDIA Tesla T4. Logs show 6 h 54 min 26 s elapsed from first training start to final export, including interruptions/restarts. Last resumed call: 2,845.04 s (47 min 25 s). Total active training time cannot be isolated; the earlier 3.5 h/epoch estimate is superseded. |

Training used Python 3.13.16, PyTorch 2.11.0+cu130 and Transformers 5.20.2, seed 42 and weight decay 0.01.

**Loss curve:** [full-run curve](docs/training_logs/loss_curves.png), [training CSV](docs/training_logs/train_metrics.csv), [validation CSV](docs/training_logs/eval_metrics.csv), and [logs/provenance](docs/training_logs/README.md). The final 1.2545 loss belongs to restored `checkpoint-16000`, not epoch 3; the curve marks it separately from epoch 3's 1.6062.

**Validation results:** final EM **67.57%**, F1 **70.39%** at threshold −3.0664; predicted-null rate 49.2%, versus 50.07% truly unanswerable. HasAns F1 is **67.08%**, NoAns accuracy **73.69%**. Without abstention, HasAns F1 would be 84.76%, but overall F1 only 42.32%.

**Checkpoint selection:** Trainer automatically retained `checkpoint-16000` (epoch approximately 1; minimum subset `eval_loss` 1.2547). On full validation, it achieved EM 63.51% / F1 66.19% at threshold −0.038. The final epoch-3 checkpoint, **`checkpoint-49410`**, achieved EM 67.57% / F1 70.39% and was exported. Although `eval_loss` rose from approximately 1.25 to 1.6, F1 improved: subset loss was a poor selection criterion here. Evidence: [final-model evaluation](docs/evaluation/final/eval_summary.json); [`checkpoint-16000` evaluation](docs/evaluation/epoch1/eval_summary.json). The summaries confirm a 4.20-point F1 gain at their calibrated thresholds. Training logs confirm the loss-based selection; the final export identity is team-confirmed. See [report evidence](docs/report_evidence.json).

## 3. No-answer threshold

Evaluation selects the highest valid start-plus-end logit span across windows (`n_best=20`, maximum answer length 30 tokens). The null score is the minimum CLS start-plus-end logit score across windows. The service returns `null` when `delta = null_score - span_score > -3.0664472656250004`, or no valid span exists.

`evaluate.py` selects the threshold maximizing F1 over a validation sweep, not the five public cases. A lower threshold increases abstention, potentially rejecting correct answers; a higher threshold accepts more spans, potentially answering unsupported questions. Confidence is `sigmoid(min(abs(delta-threshold), 50))`, not a calibrated correctness probability.

At threshold 0, F1 is 69.54% (HasAns F1 74.99%, NoAns accuracy 64.10%); calibration adds **0.85 F1 points**. The [504-point sweep](docs/evaluation/final/threshold_sweep.csv) shows sampled thresholds −4.444375 through −0.9241875 within 0.5 points of best F1 ([curve and verification](docs/evaluation/final/README.md)). Comparisons reuse the selection split.

## 4. Serving architecture

FastAPI and Uvicorn run one worker on `127.0.0.1:8000`. A background startup task loads local weights/tokenizer once, checks configuration and warms up CPU inference. `/health` returns 503 until ready. Strict request validation rejects extra fields and invalid lengths/types. Evaluation's sliding windows and offsets produce literal spans; the API verifies the returned substring and paired nulls. Inference uses `eval()`, `torch.inference_mode()`, four CPU threads and batches of eight windows.

Middleware checks `/info` and `/predict` against `QA_API_TOKEN` using an exact, case-sensitive Bearer header and constant-time comparison. `/health` is public; errors use the professor's JSON contract.

Install Python 3.12 and copy the final export into `models/final_model/`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r src/requirements.txt
# Create private .env with QA_TEAM, QA_API_TOKEN, QA_MODEL_DIR,
# QA_TORCH_THREADS=4 and QA_BATCH_SIZE=8.
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --env-file .env --host 127.0.0.1 --port 8000 --workers 1
```

**Tunnel:** ngrok, `ngrok http http://127.0.0.1:8000 --url https://underwent-purge-chase.ngrok-free.dev --inspect=false`. Alternatively, `iniciar_ngrok.bat` loads `.env` and checks readiness/authentication before opening the tunnel. HTTPS tests originated on the server computer through ngrok. Separately, the team confirmed `/health` returned `{"status":"ok"}` from a phone on another Wi-Fi network; authenticated remote prediction remains untested.

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? | Local HTTP latency | HTTPS latency |
|---|---|---|---|---|---|---|
| sample-01 | `16` or `16 attention heads` | `16` | 383 | Yes | 0.0845 s | 0.288 s |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes | 0.0774 s | 0.239 s |
| sample-03 | `null` | `null` | `null` | Yes | 0.0714 s | 0.240 s |
| sample-04 | `null` | `null` | `null` | Yes | 0.0902 s | 0.242 s |
| sample-05 | `50,257` or `50,257 tokens` | `50,257` | 3157 | Yes | 0.6955 s | 0.885 s |

Local and HTTPS runs passed **5/5**, with HTTP 200, valid spans, paired nulls, scores and IDs. Earlier notebook failures included sample-01 and sample-04; public successes do not establish hidden-set accuracy.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | Weights: 265,470,032 bytes (265.47 MB); complete export: 266,183,844 bytes |
| Mean latency per `/predict` request on your laptop (CPU) | 0.2038 s; maximum 0.6955 s; one sequential pass of five cases over loopback HTTP |
| Laptop CPU and RAM | AMD Ryzen 5 8645HS, 6 cores/12 threads; 15.27 GiB OS-visible RAM |
| HTTPS, five public cases | Mean 0.379 s; maximum 0.885 s; 5/5 correct |
| HTTPS, 50 sequential requests | Five public cases repeated ten times: 20 short answerable, 20 unanswerable, 10 long-context requests; 50/50 correct |
| HTTPS load latency | Total 19.272 s; mean 0.385 s; maximum 1.054 s; no request exceeded 5 s |

Local runtime: Python 3.12.10, PyTorch 2.14.1+cpu, Transformers 5.20.2. Model loading plus warmup took 1.326 s; full startup to readiness took 9.267 s. HTTPS met the 5 s individual and 150 s total limits for this repeated public workload, not necessarily hidden questions or other networks. Measurements and provenance are in [report evidence](docs/report_evidence.json).

## 7. Problems encountered and how you fixed them

1. **Problem:** sample-05 has 649 context tokens, exceeding the model's 512-token capacity. **Cause:** one input cannot contain the full context. **Fix:** overlapping 384-token windows, stride 128, and original character offsets; three windows recovered the answer at offset 3157. **Commits:** `f5af3ab` (evaluation), `0e556af` (service reuse).
2. **Problem:** Trainer selected `checkpoint-16000`, despite epoch 3 achieving higher full-validation F1. **Cause:** minimum subset loss did not rank checkpoints by the final metric. **Fix:** compare full-validation F1 and export `checkpoint-49410` (Section 2). **Documentation commit:** `e64e08e` records selection evidence; the manual weight export is not a code change.

## 8. Decoder investigation

For the observed abstention failures, a [T5-style text-to-text model](https://arxiv.org/abs/1910.10683) could be trained to emit either an answer or an explicit no-answer marker. This is a proposed alternative, not a tested fix: generation does not inherently guarantee grounding, and exact spans/offsets would still need enforcement. Autoregressive decoding adds work; CPU latency and model size would require measurement.

For sample-05's length problem, [Longformer-Encoder-Decoder](https://arxiv.org/abs/2004.05150) offers sparse long-context encoding plus generation. It could combine evidence across a longer input, but adds a decoder, memory requirements and span-alignment checks. Our measured three-window encoder solution already answers this case; neither alternative was implemented or benchmarked.

## 9. Known limitations

English extractive QA only; no synthesized, non-contiguous or unsupported yes/no answers. The API rejects inputs beyond 5,000 context characters or 500 question characters. Within those character limits, an overlong tokenized question is shortened to preserve context coverage and stride. Spans are limited to 30 tokens; inference is serialized, so concurrent requests can queue. Hidden-set performance and authenticated predictions from an independent network remain unverified.
