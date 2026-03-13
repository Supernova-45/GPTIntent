#!/usr/bin/env python3
"""Blend tabular model predictions using validation-selected weights."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_predictions(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _align_split(
    payload: Dict[str, object],
    split: str,
) -> Tuple[List[str], List[int], List[float]]:
    s = payload[split]
    sample_ids = [str(x) for x in s["sample_id"]]
    y_true = [int(x) for x in s["y_true"]]
    y_score = [float(x) for x in s["y_score"]]
    return sample_ids, y_true, y_score


def _find_best_threshold_for_f1(y_true, y_score) -> float:
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y_true, y_score)
    if len(t) == 0:
        return 0.5
    f1 = (2 * p * r) / (p + r + 1e-12)
    return float(t[int(np.argmax(f1[:-1]))])


def _eval_metrics(y_true, y_score, threshold: float) -> Dict[str, float]:
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

    y_pred = (np.array(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }


def _generate_weight_vectors(n: int, step: float) -> List[List[float]]:
    if n < 2:
        return [[1.0]]
    units = int(round(1.0 / step))
    out = []
    for combo in itertools.product(range(units + 1), repeat=n):
        if sum(combo) != units:
            continue
        out.append([c / units for c in combo])
    return out


def _score_blend(y_true, scores_by_model, weights):
    import numpy as np

    score = np.zeros_like(np.asarray(next(iter(scores_by_model.values()))), dtype=float)
    for w, (_, s) in zip(weights, scores_by_model.items()):
        score += w * np.asarray(s, dtype=float)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend model predictions with validation-selected weights.")
    parser.add_argument(
        "--pred-dir",
        default="artifacts/reports/predictions",
        help="Directory with <model>_predictions.json files.",
    )
    parser.add_argument(
        "--models",
        default="xgboost,lightgbm,random_forest",
        help="Comma-separated model names to blend.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.05,
        help="Grid-search step for blend weights (e.g., 0.05).",
    )
    parser.add_argument("--output", default="artifacts/reports/blend_metrics.json")
    args = parser.parse_args()

    import numpy as np

    from analysis.common import save_json

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        raise ValueError("Need at least 2 models to blend.")

    payloads = {}
    for m in models:
        path = Path(args.pred_dir) / f"{m}_predictions.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        payloads[m] = _load_predictions(str(path))

    ref_model = models[0]
    ref_val_ids, y_val, _ = _align_split(payloads[ref_model], "val")
    ref_test_ids, y_test, _ = _align_split(payloads[ref_model], "test")

    val_scores_by_model: Dict[str, List[float]] = {}
    test_scores_by_model: Dict[str, List[float]] = {}

    for m in models:
        val_ids, yv_m, val_scores = _align_split(payloads[m], "val")
        test_ids, yt_m, test_scores = _align_split(payloads[m], "test")
        if val_ids != ref_val_ids or test_ids != ref_test_ids:
            raise ValueError(f"Sample ID ordering mismatch for model {m}.")
        if yv_m != y_val or yt_m != y_test:
            raise ValueError(f"Label mismatch for model {m}.")
        val_scores_by_model[m] = val_scores
        test_scores_by_model[m] = test_scores

    # Baseline best single-model on val AUPRC
    best_single_name = None
    best_single_val_auprc = -1.0
    best_single_test_metrics = None
    for m in models:
        val_metric = _eval_metrics(y_val, val_scores_by_model[m], threshold=0.5)["auprc"]
        if val_metric > best_single_val_auprc:
            best_single_val_auprc = val_metric
            best_single_name = m
            t = _find_best_threshold_for_f1(y_val, val_scores_by_model[m])
            best_single_test_metrics = _eval_metrics(y_test, test_scores_by_model[m], threshold=t)

    candidates = _generate_weight_vectors(len(models), args.weight_step)
    best = None
    for w in candidates:
        val_blend = _score_blend(y_val, {m: val_scores_by_model[m] for m in models}, w)
        val_auprc = _eval_metrics(y_val, val_blend, threshold=0.5)["auprc"]
        if best is None or val_auprc > best["val_auprc"]:
            best = {"weights": w, "val_auprc": float(val_auprc)}

    assert best is not None
    w = best["weights"]
    val_blend = _score_blend(y_val, {m: val_scores_by_model[m] for m in models}, w)
    test_blend = _score_blend(y_test, {m: test_scores_by_model[m] for m in models}, w)
    t_blend = _find_best_threshold_for_f1(y_val, val_blend)
    blend_test_metrics = _eval_metrics(y_test, test_blend, threshold=t_blend)

    deltas = {
        k: float(blend_test_metrics[k] - best_single_test_metrics[k])
        for k in ["auprc", "roc_auc", "f1", "accuracy", "precision", "recall"]
    }

    payload = {
        "models": models,
        "weight_step": float(args.weight_step),
        "best_single_model": best_single_name,
        "best_single_test_metrics": best_single_test_metrics,
        "best_blend": {
            "weights": {m: float(wi) for m, wi in zip(models, w)},
            "val_auprc": float(best["val_auprc"]),
            "test_metrics": blend_test_metrics,
        },
        "deltas_vs_best_single": deltas,
        "num_weight_candidates": len(candidates),
    }
    save_json(args.output, payload)
    print(f"Blend metrics saved: {args.output}", flush=True)
    print("Best weights:", payload["best_blend"]["weights"], flush=True)
    print("Delta AUPRC:", f"{deltas['auprc']:+.6f}", flush=True)


if __name__ == "__main__":
    main()
