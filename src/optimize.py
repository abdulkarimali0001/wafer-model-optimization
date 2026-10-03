"""Compress the wafer defect model and measure size, speed and accuracy.

Variants compared on the same 5,701 held-out test wafers:
  1. PyTorch FP32           the original model
  2. ONNX FP32              same weights, exported to ONNX Runtime (portable, used on edge devices)
  3. ONNX INT8 (static)     weights AND activations in 8-bit, calibrated on 512 training wafers
  4. Student FP32 (ONNX)    4x smaller CNN trained by knowledge distillation (src/distill.py)
  5. Student INT8 (ONNX)    distillation + quantization combined

Usage:  python src/optimize.py --data ../wafer-defect-classifier/data/Wafer_Map_Datasets.npz
Writes results/comparison.csv, results/comparison.json, results/tradeoff.png
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from sklearn.metrics import f1_score

from data import load_npz, one_hot_maps, stratified_split
from model import WaferNet, count_params

ROOT = Path(__file__).resolve().parent.parent
M, R = ROOT / "models", ROOT / "results"
torch.set_num_threads(1)


def export_onnx(model, path):
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, 52, 52), str(path), input_names=["wafer"], output_names=["logits"],
                      dynamic_axes={"wafer": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)


class Calib(CalibrationDataReader):
    def __init__(self, x):
        self.it = iter([{"wafer": x[i:i + 32]} for i in range(0, len(x), 32)])

    def get_next(self):
        return next(self.it, None)


def quantize(fp32_path, int8_path, calib_x):
    pre = fp32_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    quantize_static(str(pre), str(int8_path), Calib(calib_x), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()


def session(path):
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def evaluate(predict, x, y):
    probs = np.concatenate([predict(x[i:i + 256]) for i in range(0, len(x), 256)])
    pred = (probs >= 0.5).astype(int)
    present = np.where(y.sum(0) > 0)[0]
    return float(f1_score(y, pred, average="macro", labels=present, zero_division=0)), float((pred == y).all(1).mean())


def latency_ms(predict, x1, n=500):
    for _ in range(30):
        predict(x1)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); predict(x1); t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


def throughput(predict, xb, reps=10):
    predict(xb); t0 = time.perf_counter()
    for _ in range(reps):
        predict(xb)
    return reps * len(xb) / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); args = ap.parse_args()
    R.mkdir(exist_ok=True)
    maps, labels = load_npz(args.data)
    tr, _, te = stratified_split(labels)
    xte, yte = one_hot_maps(maps[te]), labels[te]
    calib = one_hot_maps(maps[np.random.default_rng(0).choice(tr, 512, replace=False)])

    sig = lambda z: 1 / (1 + np.exp(-z))
    teacher = WaferNet(); teacher.load_state_dict(torch.load(M / "wafernet_fp32.pt")); teacher.eval()
    student = WaferNet(width=16); student.load_state_dict(torch.load(M / "wafernet_student.pt")); student.eval()

    export_onnx(teacher, M / "wafernet_fp32.onnx"); quantize(M / "wafernet_fp32.onnx", M / "wafernet_int8.onnx", calib)
    export_onnx(student, M / "student_fp32.onnx"); quantize(M / "student_fp32.onnx", M / "student_int8.onnx", calib)

    def torch_pred(x):
        with torch.inference_mode():
            return torch.sigmoid(teacher(torch.from_numpy(x))).numpy()

    def ort_pred(path):
        s = session(path)
        return lambda x: sig(s.run(None, {"wafer": x})[0])

    variants = [
        ("PyTorch FP32", torch_pred, M / "wafernet_fp32.pt", count_params(teacher)),
        ("ONNX FP32", ort_pred(M / "wafernet_fp32.onnx"), M / "wafernet_fp32.onnx", count_params(teacher)),
        ("ONNX INT8 (static quantization)", ort_pred(M / "wafernet_int8.onnx"), M / "wafernet_int8.onnx", count_params(teacher)),
        ("Student FP32 (distilled, ONNX)", ort_pred(M / "student_fp32.onnx"), M / "student_fp32.onnx", count_params(student)),
        ("Student INT8 (distilled + quantized)", ort_pred(M / "student_int8.onnx"), M / "student_int8.onnx", count_params(student)),
    ]
    rows = []
    for name, pred, path, params in variants:
        f1, em = evaluate(pred, xte, yte)
        rows.append({"variant": name, "parameters": params, "size_mb": round(os.path.getsize(path) / 1e6, 3),
                     "latency_ms_batch1": round(latency_ms(pred, xte[:1]), 3),
                     "throughput_wafers_per_s": round(throughput(pred, xte[:256]), 1),
                     "test_macro_f1": round(f1, 4), "test_exact_match": round(em, 4)})
        print(rows[-1], flush=True)

    base = rows[0]
    for r in rows:
        r["size_vs_original"] = f"{base['size_mb'] / r['size_mb']:.1f}x smaller"
        r["speedup_vs_original"] = f"{base['latency_ms_batch1'] / r['latency_ms_batch1']:.1f}x"
    (R / "comparison.json").write_text(json.dumps(rows, indent=2))
    keys = list(rows[0])
    (R / "comparison.csv").write_text(",".join(keys) + "\n" + "".join(",".join(str(r[k]) for k in keys) + "\n" for r in rows))

    from plot import main as plot
    plot()

if __name__ == "__main__":
    main()
