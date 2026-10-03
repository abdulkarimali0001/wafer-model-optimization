"""Knowledge distillation: train a 4x smaller student CNN to copy the big model.

The student (width 16, about 295K parameters) learns from two signals:
  - the true labels (normal BCE loss), and
  - the teacher's probabilities ("soft targets"), which carry extra information,
    e.g. "this Loc cluster looks a bit like a Scratch".

Usage:  python src/distill.py --data ../wafer-defect-classifier/data/Wafer_Map_Datasets.npz --epochs 10
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from data import WaferDataset, load_npz, stratified_split
from model import WaferNet, count_params

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the teacher's soft targets")
    args = ap.parse_args()
    torch.manual_seed(0)

    maps, labels = load_npz(args.data)
    tr, va, _ = stratified_split(labels)  # same split as the training project
    train_dl = DataLoader(WaferDataset(maps[tr], labels[tr], augment=True), batch_size=128, shuffle=True)
    val_dl = DataLoader(WaferDataset(maps[va], labels[va]), batch_size=256)

    teacher = WaferNet(); teacher.load_state_dict(torch.load(ROOT / "models" / "wafernet_fp32.pt")); teacher.eval()
    student = WaferNet(width=16)
    print(f"teacher {count_params(teacher):,} params -> student {count_params(student):,} params")

    opt = torch.optim.AdamW(student.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * len(train_dl))
    bce = nn.BCEWithLogitsLoss()
    best = -1
    for ep in range(1, args.epochs + 1):
        student.train(); t0 = time.time()
        for x, y in train_dl:
            with torch.no_grad():
                soft = torch.sigmoid(teacher(x))
            out = student(x)
            loss = (1 - args.alpha) * bce(out, y) + args.alpha * bce(out, soft)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        student.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in val_dl:
                ps.append(torch.sigmoid(student(x))); ys.append(y)
        p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
        f1 = f1_score(y, p >= 0.5, average="macro", labels=np.where(y.sum(0) > 0)[0], zero_division=0)
        print(f"epoch {ep:2d} | val macro-F1 {f1:.4f} | {time.time() - t0:.0f}s", flush=True)
        if f1 > best:
            best = f1; torch.save(student.state_dict(), ROOT / "models" / "wafernet_student.pt")
    print(f"best val macro-F1 {best:.4f}")


if __name__ == "__main__":
    main()
