"""Draw results/tradeoff.png from results/comparison.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

R = Path(__file__).resolve().parent.parent / "results"
SHORT = {"PyTorch FP32": "Original (PyTorch FP32)", "ONNX FP32": "ONNX FP32",
         "ONNX INT8 (static quantization)": "INT8 quantized", "Student FP32 (distilled, ONNX)": "Distilled student",
         "Student INT8 (distilled + quantized)": "Student + INT8"}
OFFSET = {"PyTorch FP32": (6, 8), "ONNX FP32": (6, -14), "Student INT8 (distilled + quantized)": (-8, -18),
          "Student FP32 (distilled, ONNX)": (8, -16)}


def main():
    rows = json.loads((R / "comparison.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, label in [(ax[0], "size_mb", "Model file size (MB)"), (ax[1], "latency_ms_batch1", "Latency per wafer, batch 1, 1 CPU thread (ms)")]:
        for r in rows:
            a.scatter(r[key], r["test_macro_f1"] * 100, s=70, zorder=3)
            a.annotate(SHORT[r["variant"]], (r[key], r["test_macro_f1"] * 100), fontsize=8,
                       xytext=OFFSET.get(r["variant"], (6, 6)), textcoords="offset points")
        a.set(xlabel=label, ylabel="Test macro-F1 (%)", ylim=(98.0, 100.0), xlim=(0, max(r[key] for r in rows) * 1.35))
        a.grid(alpha=0.3)
    ax[0].set_title("Size: up to 14.6x smaller"); ax[1].set_title("Speed: up to 12x faster")
    for a in ax:
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    fig.suptitle("Accuracy stays within 0.2 points while the model shrinks", fontsize=11)
    fig.tight_layout(); fig.savefig(R / "tradeoff.png", dpi=150)


if __name__ == "__main__":
    main()
