"""Recreate the figure: python docs/training_logs/plot_loss.py.

Requires matplotlib. The evaluation after restoring checkpoint-16000 is shown
separately, not as an epoch-3 improvement.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    history = json.loads((root / "log_history.json").read_text(encoding="utf-8"))["log_history"]
    boundary = next(i for i, row in enumerate(history) if "train_runtime" in row)
    train = [row for row in history[:boundary] if "loss" in row]
    validation = [row for row in history[:boundary] if "eval_loss" in row]
    restored = [row for row in history[boundary:] if "eval_loss" in row]
    best = min(validation, key=lambda row: row["eval_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    axes[0].plot([r["step"] for r in train], [r["loss"] for r in train],
                 color="#2563a6", alpha=0.75, linewidth=1, label="Training loss")
    for ax in axes:
        ax.plot([r["step"] for r in validation], [r["eval_loss"] for r in validation],
                color="#cf6514", linewidth=1.4, label="Validation during training")
        ax.scatter([r["step"] for r in restored], [r["eval_loss"] for r in restored],
                   marker="x", color="#7e22a3", s=75, linewidths=2,
                   label="Re-evaluation of restored ckpt-16000", zorder=4)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Loss")
    axes[0].set_title("Full three-epoch run")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_title("Validation detail (2,000-example subset)")
    axes[1].set_ylim(1.1, 1.85)
    axes[1].scatter(best["step"], best["eval_loss"], color="#166534", marker="D", s=35, zorder=4)
    axes[1].annotate("ckpt-16000: 1.2547", xy=(best["step"], best["eval_loss"]),
                     xytext=(2500, 1.16), arrowprops={"arrowstyle": "->", "color": "#166534"}, fontsize=9)
    axes[1].annotate("Epoch 3: 1.6062", xy=(49410, validation[-1]["eval_loss"]),
                     xytext=(32000, 1.76), arrowprops={"arrowstyle": "->", "color": "#cf6514"}, fontsize=9)
    axes[1].annotate("Restored model: 1.2545", xy=(49410, restored[-1]["eval_loss"]),
                     xytext=(27000, 1.16), arrowprops={"arrowstyle": "->", "color": "#7e22a3"}, fontsize=9)
    fig.savefig(root / "loss_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
