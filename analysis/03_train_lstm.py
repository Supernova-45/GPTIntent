#!/usr/bin/env python3
"""Train LSTM model on packet size/time sequences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import time


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
    L = np.zeros((len(rows),), dtype=np.int64)
    y = np.zeros((len(rows),), dtype=np.float32)
    sample_ids: List[str] = []
    prompt_ids: List[str] = []

    for i, r in enumerate(rows):
        seq, mask = _pad_truncate(r.data_lengths, r.time_diffs, max_len)
        X[i] = np.array(seq, dtype=np.float32)
        M[i] = np.array(mask, dtype=np.float32)
        L[i] = int(min(len(r.data_lengths), max_len))
        y[i] = float(r.label)
        sample_ids.append(r.sample_id)
        prompt_ids.append(r.prompt_id)

    return X, M, L, y, sample_ids, prompt_ids


def _transform_inputs(X, M, mean=None, std=None, use_log1p: bool = False, normalize: bool = False):
    import numpy as np

    X_out = X.copy()
    mask3 = M[:, :, None] > 0

    if use_log1p:
        # Packet lengths and inter-arrival times are right-skewed.
        X_out[:, :, 0] = np.log1p(np.clip(X_out[:, :, 0], a_min=0.0, a_max=None))
        X_out[:, :, 1] = np.log1p(np.clip(X_out[:, :, 1], a_min=0.0, a_max=None))
        X_out = np.where(mask3, X_out, 0.0).astype(np.float32)

    if normalize:
        if mean is None or std is None:
            denom = float(M.sum())
            if denom <= 0:
                mean = np.zeros((2,), dtype=np.float32)
                std = np.ones((2,), dtype=np.float32)
            else:
                mean = (X_out * M[:, :, None]).sum(axis=(0, 1)) / denom
                centered = (X_out - mean[None, None, :]) * M[:, :, None]
                var = (centered ** 2).sum(axis=(0, 1)) / denom
                std = np.sqrt(var + 1e-6)
        X_out = np.where(mask3, (X_out - mean[None, None, :]) / std[None, None, :], 0.0).astype(np.float32)

    return X_out, mean, std


def _evaluate(y_true, y_score, threshold: float, precision_ks, bootstrap_samples: int, seed: int) -> Dict[str, object]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from analysis.common import bootstrap_metric_ci, precision_at_k

    y_pred = (np.array(y_score) >= threshold).astype(int)
    auprc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
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
        "roc_auc": roc_auc,
        "auprc_ci95": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1": f1,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--pooling", choices=["last", "mean", "max", "attention"], default="last")
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--log1p-inputs", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-len", type=int, default=0, help="Override sequence.max_len from config when > 0")
    parser.add_argument("--train-limit", type=int, default=0, help="Optional cap on number of training rows.")
    parser.add_argument("--val-limit", type=int, default=0, help="Optional cap on number of validation rows.")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional cap on number of test rows.")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--predictions-path", default="")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate a saved model.")
    parser.add_argument("--load-model-path", default="", help="Path to a saved model for --eval-only mode.")
    args = parser.parse_args()

    _require_torch()

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import average_precision_score

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.random_seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device(args.device)

    rows = _load_sequence_rows(config)
    _assign_splits(rows, config.split)

    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    test_rows = [r for r in rows if r.split == "test"]

    def _cap_rows(items, limit: int):
        if limit <= 0 or len(items) <= limit:
            return items
        import numpy as np

        rng = np.random.default_rng(config.training.random_seed)
        idx = rng.choice(len(items), size=limit, replace=False)
        idx = sorted(int(i) for i in idx.tolist())
        return [items[i] for i in idx]

    train_rows = _cap_rows(train_rows, args.train_limit)
    val_rows = _cap_rows(val_rows, args.val_limit)
    test_rows = _cap_rows(test_rows, args.test_limit)

    max_len = args.max_len if args.max_len > 0 else config.sequence.max_len
    print(
        f"[lstm] device={device} max_len={max_len} "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"batch_size={args.batch_size} epochs={args.epochs}",
        flush=True,
    )
    X_train, M_train, L_train, y_train, train_sample_ids, train_prompt_ids = _build_arrays(train_rows, max_len)
    X_val, M_val, L_val, y_val, val_sample_ids, val_prompt_ids = _build_arrays(val_rows, max_len)
    X_test, M_test, L_test, y_test, test_sample_ids, test_prompt_ids = _build_arrays(test_rows, max_len)

    norm_mean = None
    norm_std = None
    X_train, norm_mean, norm_std = _transform_inputs(
        X_train,
        M_train,
        use_log1p=args.log1p_inputs,
        normalize=args.normalize_inputs,
    )
    X_val, _, _ = _transform_inputs(
        X_val,
        M_val,
        mean=norm_mean,
        std=norm_std,
        use_log1p=args.log1p_inputs,
        normalize=args.normalize_inputs,
    )
    X_test, _, _ = _transform_inputs(
        X_test,
        M_test,
        mean=norm_mean,
        std=norm_std,
        use_log1p=args.log1p_inputs,
        normalize=args.normalize_inputs,
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(L_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    class LSTMClassifier(nn.Module):
        def __init__(
            self,
            input_size=2,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            pooling="last",
        ):
            super().__init__()
            self.pooling = pooling
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=(dropout if num_layers > 1 else 0.0),
                bidirectional=bidirectional,
            )
            out_dim = hidden_size * (2 if bidirectional else 1)
            if self.pooling == "attention":
                self.attn = nn.Linear(out_dim, 1)
            self.fc = nn.Linear(out_dim, 1)

        def _pool(self, out, lengths):
            bsz, seq_len, feat_dim = out.shape
            idxs = torch.arange(seq_len, device=out.device).unsqueeze(0).expand(bsz, -1)
            mask = idxs < lengths.unsqueeze(1)

            if self.pooling == "last":
                last_idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, feat_dim)
                return out.gather(1, last_idx).squeeze(1)
            if self.pooling == "mean":
                out_masked = out * mask.unsqueeze(2)
                denom = lengths.clamp(min=1).unsqueeze(1).to(out.dtype)
                return out_masked.sum(dim=1) / denom
            if self.pooling == "max":
                neg_inf = torch.full_like(out, -1e9)
                out_masked = torch.where(mask.unsqueeze(2), out, neg_inf)
                return out_masked.max(dim=1).values
            if self.pooling == "attention":
                score = self.attn(out).squeeze(2)
                score = score.masked_fill(~mask, -1e9)
                w = torch.softmax(score, dim=1).unsqueeze(2)
                return (out * w).sum(dim=1)
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        def forward(self, x, lengths):
            lengths_cpu = lengths.to(dtype=torch.long).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x.size(1),
            )
            pooled = self._pool(out, lengths.to(dtype=torch.long).clamp(min=1))
            return self.fc(pooled).squeeze(1)

    model = LSTMClassifier(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        pooling=args.pooling,
    ).to(device)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    X_val_t = torch.from_numpy(X_val).to(device)
    L_val_t = torch.from_numpy(L_val).to(device)
    y_val_np = y_val.astype(int)

    best_state = None
    best_val_auprc = -1.0
    best_epoch = 0

    if args.eval_only:
        model_path = args.load_model_path or args.model_path or f"{config.paths.models_dir}/lstm.pt"
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
            )

        bad_epochs = 0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            for xb, lb, yb in train_loader:
                xb = xb.to(device)
                lb = lb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb, lb)
                loss = criterion(logits, yb)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t, L_val_t)
                val_scores = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_auprc = float(average_precision_score(y_val_np, val_scores))

            if scheduler is not None:
                scheduler.step(val_auprc)

            if val_auprc > best_val_auprc + 1e-9:
                best_val_auprc = val_auprc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(
                f"[lstm] epoch={epoch}/{args.epochs} val_auprc={val_auprc:.4f} "
                f"best={best_val_auprc:.4f} elapsed_s={time.time()-epoch_start:.1f}",
                flush=True,
            )

            if args.early_stop_patience > 0 and epoch >= args.min_epochs and bad_epochs >= args.early_stop_patience:
                print(f"[lstm] early_stop triggered at epoch {epoch}", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

    def predict_scores(X, L):
        model.eval()
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(L))
        loader = DataLoader(ds, batch_size=max(args.batch_size, 256), shuffle=False)
        out = []
        with torch.no_grad():
            for xb, lb in loader:
                xb = xb.to(device)
                lb = lb.to(device)
                logits = model(xb, lb)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out, axis=0)

    val_scores = predict_scores(X_val, L_val)
    test_scores = predict_scores(X_test, L_test)

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
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_val_auprc),
    }

    ensure_dir(config.paths.models_dir)
    ensure_dir(config.paths.reports_dir)
    model_path = args.model_path or f"{config.paths.models_dir}/lstm.pt"
    if not args.eval_only:
        torch.save(model.state_dict(), model_path)

    payload = {
        "lstm": results,
        "model_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "bidirectional": bool(args.bidirectional),
            "pooling": args.pooling,
            "scheduler": args.scheduler,
            "early_stop_patience": args.early_stop_patience,
            "min_epochs": args.min_epochs,
            "grad_clip": args.grad_clip,
            "device": str(device),
            "max_len": max_len,
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
            "test_limit": args.test_limit,
            "log1p_inputs": bool(args.log1p_inputs),
            "normalize_inputs": bool(args.normalize_inputs),
            "normalization_mean": [float(x) for x in (norm_mean if norm_mean is not None else np.zeros(2))],
            "normalization_std": [float(x) for x in (norm_std if norm_std is not None else np.ones(2))],
        },
        "env_versions": print_env_versions(),
    }

    if args.predictions_path:
        save_json(
            args.predictions_path,
            {
                "model": "lstm",
                "train": {
                    "sample_id": train_sample_ids,
                    "prompt_id": train_prompt_ids,
                    "y_true": [int(x) for x in y_train.astype(int).tolist()],
                },
                "val": {
                    "sample_id": val_sample_ids,
                    "prompt_id": val_prompt_ids,
                    "y_true": [int(x) for x in y_val.astype(int).tolist()],
                    "y_score": [float(x) for x in val_scores.tolist()],
                },
                "test": {
                    "sample_id": test_sample_ids,
                    "prompt_id": test_prompt_ids,
                    "y_true": [int(x) for x in y_test.astype(int).tolist()],
                    "y_score": [float(x) for x in test_scores.tolist()],
                },
            },
        )

    out = args.report_path or f"{config.paths.reports_dir}/lstm_metrics.json"
    save_json(out, payload)
    print(f"LSTM metrics saved: {out}", flush=True)


if __name__ == "__main__":
    main()
