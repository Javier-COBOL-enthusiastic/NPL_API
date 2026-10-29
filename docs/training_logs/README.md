# Final training run: source logs and curve

The six original files were supplied in `logs.zip`. They are preserved byte-for-byte;
the supplied image is named `loss_curves_original.png`. No weights or credentials
are included. SHA-256 hashes are recorded in `../report_evidence.json`.

| File | Purpose |
|---|---|
| `train_metrics.csv` | Appended training metrics, including repeated steps after resumes. |
| `eval_metrics.csv` | Appended subset validation loss, including post-training evaluation. |
| `log_history.json` | Trainer history through step 49,410 / epoch 3. |
| `training_log.txt` | Startup, resume and completion timestamps; original log clock/timezone retained. |
| `final_metrics.json` | Last resumed training call and evaluation of Trainer's restored best-loss model. |
| `loss_curves_original.png` | Original figure supplied by the team, retained for traceability. |
| `loss_curves.png` | Reconstructed figure distinguishing training-time loss from the restored-model evaluation. |

## Checkpoint interpretation

The minimum training-time validation loss is **1.2546775341** at step **16,000**
(epoch 0.97146). At step **49,410**, the epoch-3 model has loss **1.6062064171**.
After training, Trainer reloads its best-loss checkpoint, `checkpoint-16000`, and
evaluates it again, obtaining **1.2545212507**. The history still labels this last
evaluation as step 49,410 / epoch 3, although its weights are from checkpoint-16000.

Therefore, deduplicating evaluation rows by step and keeping the last one hides
epoch 3's actual loss. The supplied figure shows that misleading final dip. The
reconstructed figure plots 99 evaluations during training and shows the restored
model evaluation as a separate purple cross. It uses the 988 training-loss records
in `log_history.json`; the raw CSVs remain unchanged.

`final_metrics.json` describes Trainer's automatic loss-based export, not the later
manual selection by full-validation F1. The team identified checkpoint-49410 as the
served model. Its supplied evaluation summary is in
[`../evaluation/final/eval_summary.json`](../evaluation/final/eval_summary.json);
the comparison for checkpoint-16000 is in
[`../evaluation/epoch1/eval_summary.json`](../evaluation/epoch1/eval_summary.json).
The summaries themselves do not embed checkpoint identifiers.

## Timing

- First training start: `2026-10-25 15:55:54.916`.
- Resume from checkpoint-24500: `2026-10-25 19:09:13.608`.
- Resume from checkpoint-41000: `2026-10-25 22:02:24.393`.
- Completion/export: `2026-10-25 22:50:21.195`.
- Elapsed start-to-completion interval: **24,866.279 s = 6 h 54 min 26.279 s**.
- `train_runtime` in the final metrics: **2,845.0365 s = 47 min 25.0365 s**, for
  the final resumed call, not all three epochs.

The elapsed interval includes interruptions, restart/setup overhead, validation,
checkpointing and export. The logs do not identify the stop time of each interrupted
session, so they do not establish total active training time or a measured time per
epoch. The team's earlier 3.5-hours-per-epoch estimate is superseded by this more
precise account. No timezone conversion has been assumed.

## Reproduce the curve

From the repository root, with Matplotlib installed in the local environment:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
.\.venv\Scripts\python.exe docs/training_logs/plot_loss.py
```

This reads saved metrics only; it does not run training, inference or evaluation.
