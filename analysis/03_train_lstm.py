#!/usr/bin/env python3
"""Train LSTM model on packet size/time sequences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as ex:
        raise RuntimeError("PyTorch is required for LSTM training. Install with: pip install torch") from ex


@dataclass
class SequenceRow:
    sample_id: str
    prompt_id: str
    label: int
    split: str
    data_lengths: List[float]
    time_diffs: List[float]


def _load_sequence_rows(config):
    import glob
    import json

    from analysis.common import load_prompt_sets, sha1_text

    files = sorted(glob.glob(config.paths.data_glob))
    if not files:
        raise FileNotFoundError(f"No shard files found: {config.paths.data_glob}")

    pos, neg = load_prompt_sets(config.paths.prompts_path)
    rows: List[SequenceRow] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for item in arr:
            prompt = item.get("prompt")
            if prompt in pos:
                label = 1
            elif prompt in neg:
                label = 0
            else:
                continue

            dl = item.get("data_lengths") or []
            td = item.get("time_diffs") or []
            if not isinstance(dl, list) or not isinstance(td, list):
                continue
            if len(dl) == 0 or len(dl) != len(td):
                continue

            trial = int(item.get("trial", 0))
            ts = float(item.get("timestamp", 0.0))
            sample_id = sha1_text(f"{prompt}|{trial}|{ts}")
            prompt_id = sha1_text(prompt)
            rows.append(
                SequenceRow(
                    sample_id=sample_id,
                    prompt_id=prompt_id,
                    label=label,
                    split="",
                    data_lengths=[float(x) for x in dl],
                    time_diffs=[float(x) for x in td],
                )
            )
    return rows


def _assign_splits(rows: List[SequenceRow], split_cfg):
    from analysis.common import assign_split, group_stratified_split

    split_groups = group_stratified_split(
        [r.prompt_id for r in rows],
        [r.label for r in rows],
        split_cfg,
    )
    for r in rows:
        r.split = assign_split(r.prompt_id, split_groups)


def _pad_truncate(dl: List[float], td: List[float], max_len: int) -> Tuple[List[List[float]], List[float]]:
    n = min(len(dl), max_len)
    seq = [[dl[i], td[i]] for i in range(n)]
    mask = [1.0] * n
    if n < max_len:
        pad_count = max_len - n
        seq.extend([[0.0, 0.0]] * pad_count)
        mask.extend([0.0] * pad_count)
    return seq, mask


def _build_arrays(rows: List[SequenceRow], max_len: int):
    import numpy as np

    X = np.zeros((len(rows), max_len, 2), dtype=np.float32)
    M = np.zeros((len(rows), max_len), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.float32)

    for i, r in enumerate(rows):
        seq, mask = _pad_truncate(r.data_lengths, r.time_diffs, max_len)
        X[i] = np.array(seq, dtype=np.float32)
        M[i] = np.array(mask, dtype=np.float32)
        y[i] = float(r.label)

    return X, M, y


def _evaluate(y_true, y_score, threshold: float, precision_ks, bootstrap_samples: int, seed: int) -> Dict[str, object]:
    import numpy as np
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score

    from analysis.common import bootstrap_metric_ci, precision_at_k

    y_pred = (np.array(y_score) >= threshold).astype(int)
    auprc = float(average_precision_score(y_true, y_score))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()
    p_at_k = {f"p@{int(k*100)}": float(precision_at_k(y_true, y_score, k)) for k in precision_ks}

    auprc_ci = bootstrap_metric_ci(
        y_true,
        y_score,
        lambda yt, ys: float(average_precision_score(yt, ys)),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )

    return {
        "threshold": float(threshold),
        "auprc": auprc,
        "auprc_ci95": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1": f1,
        "accuracy": acc,
        "confusion_matrix": cm,
        "precision_at_k": p_at_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM baseline")
    parser.add_argument("--config", default="analysis/config.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    _require_torch()

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from analysis.common import (
        ensure_dir,
        find_threshold_for_best_f1,
        find_threshold_for_target_precision,
        load_config,
        print_env_versions,
        save_json,
        set_seed,
    )

    config = load_config(args.config)
    set_seed(config.training.random_seed)
    torch.manual_seed(config.training.random_seed)

    rows = _load_sequence_rows(config)
    _assign_splits(rows, config.split)

    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    test_rows = [r for r in rows if r.split == "test"]

    X_train, M_train, y_train = _build_arrays(train_rows, config.sequence.max_len)
    X_val, M_val, y_val = _build_arrays(val_rows, config.sequence.max_len)
    X_test, M_test, y_test = _build_arrays(test_rows, config.sequence.max_len)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(M_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=2, hidden_size=64, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x, mask):
            out, _ = self.lstm(x)
            lengths = mask.sum(dim=1).long().clamp(min=1)
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
            last = out.gather(1, idx).squeeze(1)
            return self.fc(last).squeeze(1)

    model = LSTMClassifier(hidden_size=args.hidden_size, num_layers=args.num_layers)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for _ in range(args.epochs):
        for xb, mb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    def predict_scores(X, M):
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X), torch.from_numpy(M))
            return torch.sigmoid(logits).cpu().numpy()

    val_scores = predict_scores(X_val, M_val)
    test_scores = predict_scores(X_test, M_test)

    t_best_f1 = find_threshold_for_best_f1(y_val, val_scores)
    t_high_prec = find_threshold_for_target_precision(y_val, val_scores, target_precision=0.90)

    results = {
        "threshold_best_f1": _evaluate(
            y_test,
            test_scores,
            threshold=t_best_f1,
            precision_ks=config.training.precision_at_k,
            bootstrap_samples=config.training.bootstrap_samples,
            seed=config.training.random_seed,
        ),
        "threshold_target_precision_0.90": _evaluate(
            y_test,
            test_scores,
            threshold=t_high_prec,
            precision_ks=config.training.precision_at_k,
            bootstrap_samples=config.training.bootstrap_samples,
            seed=config.training.random_seed,
        ),
        "shape": {
            "train": [int(x) for x in X_train.shape],
            "val": [int(x) for x in X_val.shape],
            "test": [int(x) for x in X_test.shape],
        },
    }

    ensure_dir(config.paths.models_dir)
    ensure_dir(config.paths.reports_dir)
    torch.save(model.state_dict(), f"{config.paths.models_dir}/lstm.pt")

    payload = {
        "lstm": results,
        "model_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "max_len": config.sequence.max_len,
        },
        "env_versions": print_env_versions(),
    }

    out = f"{config.paths.reports_dir}/lstm_metrics.json"
    save_json(out, payload)
    print(f"LSTM metrics saved: {out}")


if __name__ == "__main__":
    main()
