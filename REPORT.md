# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | [COMPLETAR] (`team-local` was a temporary test value) |
| Members | Dennys Jefferson Molina Abrego; Javier [COMPLETAR: full name]; [COMPLETAR: remaining members] |
| Repository URL | https://github.com/Javier-COBOL-enthusiastic/NPL_API |
| `submission` commit hash | [COMPLETAR] |
| Public service URL used at the dry run | [COMPLETAR] |
| AI tools used (which, and for what) | Codex: API, authentication, tests, Windows/ngrok launcher, local verification and report drafting. [COMPLETAR: other tools used by the team, if any] |

## 1. Model choice and justification

**Base checkpoint:** `distilbert/distilbert-base-uncased`, fine-tuned with a question-answering head, not a checkpoint already trained on SQuAD. Its six-layer encoder provides a practical CPU serving footprint: 66.36 million parameters and a measured 0.204 s mean local request time. The final validation F1 is 70.39%; this is a quality/size/latency trade-off, not evidence of superiority over untested alternatives.

**Alternatives considered:** [COMPLETAR: at least one model actually considered and the team's reason for rejecting it].

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Full `rajpurkar/squad_v2` train: 130,319 examples; 131,754 tokenized windows |
| Validation data (what you validated on, how many examples) | 2,000 examples for training-time loss; full 11,873-example validation split for final EM/F1 and threshold calibration |
| Epochs | 3 |
| Learning rate and schedule (warmup?) | `3e-5`; linear schedule (Trainer default); 10% warmup, 4,941 steps |
| Batch size | 8 per device; gradient accumulation 1 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | Google Colab, NVIDIA Tesla T4; total time across resumed sessions: [COMPLETAR] |

Training used Python 3.13.16, PyTorch 2.11.0+cu130 and Transformers 5.20.2, seed 42 and weight decay 0.01. The script supports resuming model, optimizer and scheduler state.

**Loss curve:** the [notebook](src/training/extractive_qa_colab.ipynb) embeds a curve, but its saved outputs describe an earlier run. [COMPLETAR: repository link to the complete final-run loss curve and logs].

**Validation results:** final EM **67.5735%**, F1 **70.3899%**, from the exported calibration metadata in [report evidence](docs/report_evidence.json). Final HasAns/NoAns breakdown: [COMPLETAR]. The notebook's older F1 of 60.03% must not be substituted. Javier confirmed selection of the third-epoch checkpoint by F1; [COMPLETAR: exact checkpoint identifier and comparison evidence]. The script itself automatically selects by `eval_loss`, so the subsequent F1-based export needs that provenance.

## 3. No-answer threshold

For each context, evaluation selects the highest valid start-plus-end logit span across all windows (`n_best=20`, maximum answer length 30 tokens). The null score is the minimum CLS start-plus-end logit score across windows. The service returns `null` when `delta = null_score - span_score > -3.0664472656250004`, or no valid span exists.

`evaluate.py` selects the threshold maximizing F1 over a validation sweep, not the five public cases. A lower threshold increases abstention, potentially rejecting correct answers; a higher threshold accepts more spans, potentially answering unsupported questions. Confidence is `sigmoid(min(abs(delta-threshold), 50))`, not a calibrated correctness probability.

## 4. Serving architecture

FastAPI and Uvicorn run one worker on `127.0.0.1:8000`. A background startup task loads local weights/tokenizer once, checks configuration and warms up CPU inference. `/health` returns 503 until ready. Strict request validation rejects extra fields and invalid lengths/types. Evaluation's sliding windows and offsets produce literal spans; the API verifies the returned substring and paired nulls. Inference uses `eval()`, `torch.inference_mode()`, four CPU threads and batches of eight windows.

Middleware checks `/info` and `/predict` against `QA_API_TOKEN` using an exact, case-sensitive Bearer header and constant-time comparison. `/health` is public; errors use the professor's JSON contract.

From a clean Windows checkout, install Python 3.12 and copy the six-file final export into `models/final_model/`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r src/requirements.txt
# Create private .env with QA_TEAM, QA_API_TOKEN, QA_MODEL_DIR,
# QA_TORCH_THREADS=4 and QA_BATCH_SIZE=8.
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --env-file .env --host 127.0.0.1 --port 8000 --workers 1
```

**Tunnel:** ngrok, `ngrok http http://127.0.0.1:8000 --url https://YOUR-ASSIGNED-DOMAIN --inspect=false`. Alternatively, `iniciar_ngrok.bat` loads `.env`, starts the API, checks readiness/authentication, then opens the configured `NGROK_DOMAIN`. Actual dry-run connectivity and timing: [COMPLETAR].

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? | Local HTTP latency |
|---|---|---|---|---|---|
| sample-01 | `16` or `16 attention heads` | `16` | 383 | Yes | 0.0845 s |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes | 0.0774 s |
| sample-03 | `null` | `null` | `null` | Yes | 0.0714 s |
| sample-04 | `null` | `null` | `null` | Yes | 0.0902 s |
| sample-05 | `50,257` or `50,257 tokens` | `50,257` | 3157 | Yes | 0.6955 s |

The final local run passed **5/5**, with valid spans, scores and IDs. There were no wrong answers in this run. Earlier notebook failures are discussed below; these five cases do not establish hidden-set accuracy.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | Weights: 265,470,032 bytes (265.47 MB); complete export: 266,183,844 bytes |
| Mean latency per `/predict` request on your laptop (CPU) | 0.2038 s; maximum 0.6955 s; one sequential pass of five cases over loopback HTTP |
| Laptop CPU and RAM | AMD Ryzen 5 8645HS, 6 cores/12 threads; 15.27 GiB OS-visible RAM |

Local runtime: Python 3.12.10, PyTorch 2.14.1+cpu, Transformers 5.20.2. Model loading plus warmup took 1.326 s; full startup to readiness took 9.267 s. All five requests were below 5 s. The 150 s cap for 50 hidden questions and tunnel overhead remain unverified. Measurements are preserved in [report evidence](docs/report_evidence.json).

## 7. Problems encountered and how you fixed them

1. **Problem:** sample-05 has 649 context tokens, exceeding the model's 512-token capacity. **Cause:** one input cannot contain the full context. **Fix:** overlapping 384-token windows, stride 128, and original character offsets; three windows recovered the answer at offset 3157. **Commits:** `f5af3ab` (evaluation), `0e556af` (service reuse).
2. **Problem:** the earlier notebook returned null for sample-01 and an unsupported span for sample-04. **Cause:** incorrect answer/no-answer decisions; [COMPLETAR: confirmed contribution of weights versus threshold]. **Fix:** the final export and its recalibrated threshold passed both cases locally. **Commit/provenance:** [COMPLETAR: checkpoint-selection record and associated code/report commit; weights stay outside Git].

## 8. Decoder investigation

For the observed abstention failures, a [T5-style text-to-text model](https://arxiv.org/abs/1910.10683) could be trained to emit either an answer or an explicit no-answer marker. This is a proposed alternative, not a tested fix: generation does not inherently guarantee grounding, and exact spans/offsets would still need enforcement. Autoregressive decoding adds work; CPU latency and model size would require measurement.

For sample-05's length problem, [Longformer-Encoder-Decoder](https://arxiv.org/abs/2004.05150) offers sparse long-context encoding plus generation. It could combine evidence across a longer input, but adds a decoder, memory requirements and span-alignment checks. Our measured three-window encoder solution already answers this case; neither alternative was implemented or benchmarked.

## 9. Known limitations

English extractive QA only; no synthesized, non-contiguous or unsupported yes/no answers. The API rejects inputs beyond 5,000 context characters or 500 question characters. Within those character limits, an overlong tokenized question is shortened to preserve context coverage and stride. Spans are limited to 30 tokens; inference is serialized, so concurrent requests can queue. Five public successes do not establish general accuracy. Official team identity, final training evidence and public dry-run verification remain [COMPLETAR].
