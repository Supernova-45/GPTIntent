#!/usr/bin/env python3
"""Build CI summaries and provider-comparison plot from saved predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DISPLAY_NAME = {
    "logreg": "LogReg",
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "blend": "Blend",
    "lstm": "LSTM",
}


def _load_json(path: str):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate(y_true, y_score, threshold, precision_ks, bootstrap_samples: int, seed: int) -> Dict[str, object]:
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

    y_true_np = np.asarray(y_true, dtype=int)
    y_score_np = np.asarray(y_score, dtype=float)
    y_pred = (y_score_np >= threshold).astype(int)

    auprc = float(average_precision_score(y_true_np, y_score_np))
    f1 = float(f1_score(y_true_np, y_pred))
    p_at_k = {f"p@{int(k*100)}": float(precision_at_k(y_true_np, y_score_np, k)) for k in precision_ks}

    auprc_ci = bootstrap_metric_ci(
        y_true_np,
        y_score_np,
        lambda yt, ys: float(average_precision_score(yt, ys)),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )
    f1_ci = bootstrap_metric_ci(
        y_true_np,
        y_score_np,
        lambda yt, ys: float(f1_score(yt, (ys >= threshold).astype(int))),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )
    p_at_k_ci = {}
    for k in precision_ks:
        ci = bootstrap_metric_ci(
            y_true_np,
            y_score_np,
            lambda yt, ys, kk=k: float(precision_at_k(yt, ys, kk)),
            n_bootstrap=bootstrap_samples,
            seed=seed,
        )
        p_at_k_ci[f"p@{int(k*100)}"] = [float(ci[0]), float(ci[1])]

    return {
        "threshold": float(threshold),
        "auprc": auprc,
        "auprc_ci95": [float(auprc_ci[0]), float(auprc_ci[1])],
        "roc_auc": float(roc_auc_score(y_true_np, y_score_np)),
        "f1": f1,
        "f1_ci95": [float(f1_ci[0]), float(f1_ci[1])],
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred).tolist(),
        "precision_at_k": p_at_k,
        "precision_at_k_ci95": p_at_k_ci,
    }


def _load_dataset_summary(
    dataset_name: str,
    tabular_metrics_path: str,
    lstm_metrics_path: str,
    blend_metrics_path: str,
    pred_dir: str,
    model_names: List[str],
    precision_ks,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    import numpy as np

    tabular = _load_json(tabular_metrics_path)
    lstm_payload = _load_json(lstm_metrics_path)
    blend_payload = _load_json(blend_metrics_path)
    pred_root = Path(pred_dir)

    reports: Dict[str, object] = {}
    base_y_true = None
    base_scores_by_model: Dict[str, np.ndarray] = {}

    for model in model_names:
        pred_path = pred_root / f"{model}_predictions.json"
        if not pred_path.exists():
            continue
        pred = _load_json(str(pred_path))
        test = pred["test"]
        y_true = [int(x) for x in test["y_true"]]
        y_score = [float(x) for x in test["y_score"]]

        if base_y_true is None:
            base_y_true = y_true
        threshold = None
        if model in tabular.get("models", {}) and "threshold_best_f1" in tabular["models"][model]:
            threshold = float(tabular["models"][model]["threshold_best_f1"]["threshold"])
        elif model == "lstm":
            threshold = float(lstm_payload["lstm"]["threshold_best_f1"]["threshold"])
        if threshold is None:
            continue

        base_scores_by_model[model] = np.asarray(y_score, dtype=float)
        reports[model] = _evaluate(
            y_true,
            y_score,
            threshold=threshold,
            precision_ks=precision_ks,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )

    if "blend" in model_names and blend_payload.get("best_blend"):
        weights = blend_payload["best_blend"]["weights"]
        needed = [m for m in weights.keys() if m in base_scores_by_model]
        if len(needed) == len(weights):
            blend_scores = np.zeros_like(base_scores_by_model[needed[0]], dtype=float)
            for m, w in weights.items():
                blend_scores += float(w) * base_scores_by_model[m]
            threshold = float(blend_payload["best_blend"]["test_metrics"]["threshold"])
            reports["blend"] = _evaluate(
                base_y_true,
                blend_scores,
                threshold=threshold,
                precision_ks=precision_ks,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            reports["blend"]["weights"] = {k: float(v) for k, v in weights.items()}

    return {
        "dataset_name": dataset_name,
        "tabular_metrics_path": str(tabular_metrics_path),
        "lstm_metrics_path": str(lstm_metrics_path),
        "blend_metrics_path": str(blend_metrics_path),
        "pred_dir": str(pred_dir),
        "models": reports,
    }


def _plot_provider_drop(provider_a: Dict[str, object], provider_b: Dict[str, object], out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["logreg", "random_forest", "xgboost", "lightgbm", "blend", "lstm"]
    present = [m for m in model_order if m in provider_a["models"] and m in provider_b["models"]]
    if not present:
        return

    x = np.arange(len(present))
    width = 0.38
    a_vals = [float(provider_a["models"][m]["auprc"]) for m in present]
    b_vals = [float(provider_b["models"][m]["auprc"]) for m in present]

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    bars_a = ax.bar(x - width / 2, a_vals, width, label=provider_a["dataset_name"], color="#1f77b4")
    bars_b = ax.bar(x + width / 2, b_vals, width, label=provider_b["dataset_name"], color="#ff7f0e")

    ax.set_xticks(x, [DISPLAY_NAME.get(m, m) for m in present])
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Unpatched vs Patched Provider Performance")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower left")

    for i, m in enumerate(present):
        drop = a_vals[i] - b_vals[i]
        y = max(a_vals[i], b_vals[i]) + 0.02
        ax.text(x[i], y, f"-{drop:.3f}", ha="center", va="bottom", fontsize=8, color="#444444")

    for bars in [bars_a, bars_b]:
        for bar in bars:
            v = float(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize uncertainty and plot provider drop.")
    parser.add_argument("--dataset-a-name", default="OpenRouter (unpatched)")
    parser.add_argument("--dataset-a-tabular", default="artifacts/reports/tabular_metrics.json")
    parser.add_argument("--dataset-a-lstm", default="artifacts/reports/lstm_metrics.json")
    parser.add_argument("--dataset-a-blend", default="artifacts/reports/blend_metrics_xgb_lgbm_rf.json")
    parser.add_argument("--dataset-a-preds", default="artifacts/reports/predictions")
    parser.add_argument("--dataset-b-name", default="OpenAI (patched)")
    parser.add_argument("--dataset-b-tabular", default="artifacts/runs/data2_full_fast_lstm/reports/tabular_metrics.json")
    parser.add_argument("--dataset-b-lstm", default="artifacts/runs/data2_full_fast_lstm/reports/lstm_metrics.json")
    parser.add_argument("--dataset-b-blend", default="artifacts/runs/data2_full_fast_lstm/reports/blend_metrics.json")
    parser.add_argument("--dataset-b-preds", default="artifacts/runs/data2_full_fast_lstm/reports/predictions")
    parser.add_argument("--models", default="logreg,random_forest,xgboost,lightgbm,blend,lstm")
    parser.add_argument("--precision-ks", default="0.01,0.05,0.10")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-json", default="artifacts/reports/provider_uncertainty_summary.json")
    parser.add_argument("--output-plot", default="artifacts/reports/figures/provider_drop_side_by_side.png")
    args = parser.parse_args()

    from analysis.common import print_env_versions, save_json, set_seed

    set_seed(args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    precision_ks = tuple(float(x.strip()) for x in args.precision_ks.split(",") if x.strip())

    provider_a = _load_dataset_summary(
        dataset_name=args.dataset_a_name,
        tabular_metrics_path=args.dataset_a_tabular,
        lstm_metrics_path=args.dataset_a_lstm,
        blend_metrics_path=args.dataset_a_blend,
        pred_dir=args.dataset_a_preds,
        model_names=models,
        precision_ks=precision_ks,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    provider_b = _load_dataset_summary(
        dataset_name=args.dataset_b_name,
        tabular_metrics_path=args.dataset_b_tabular,
        lstm_metrics_path=args.dataset_b_lstm,
        blend_metrics_path=args.dataset_b_blend,
        pred_dir=args.dataset_b_preds,
        model_names=models,
        precision_ks=precision_ks,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    transfer_drop = {}
    for model in models:
        if model in provider_a["models"] and model in provider_b["models"]:
            transfer_drop[model] = {
                "dataset_a_auprc": float(provider_a["models"][model]["auprc"]),
                "dataset_b_auprc": float(provider_b["models"][model]["auprc"]),
                "delta_auprc_b_minus_a": float(
                    provider_b["models"][model]["auprc"] - provider_a["models"][model]["auprc"]
                ),
            }

    payload = {
        "settings": {
            "models": models,
            "precision_ks": [float(k) for k in precision_ks],
            "bootstrap_samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
        },
        "providers": [provider_a, provider_b],
        "provider_drop": transfer_drop,
        "output_plot": str(args.output_plot),
        "env_versions": print_env_versions(),
    }
    save_json(args.output_json, payload)
    _plot_provider_drop(provider_a, provider_b, args.output_plot)

    print(f"Provider uncertainty summary saved: {args.output_json}", flush=True)
    print(f"Provider comparison plot saved: {args.output_plot}", flush=True)


if __name__ == "__main__":
    main()
