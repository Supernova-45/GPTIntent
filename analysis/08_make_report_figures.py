#!/usr/bin/env python3
"""Generate report figures from current experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREFERRED_MODEL_ORDER = [
    "logreg",
    "svm_linear",
    "svm_rbf",
    "random_forest",
    "xgboost",
    "lightgbm",
]

DISPLAY_NAME = {
    "logreg": "LogReg",
    "svm_linear": "SVM (Linear)",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "blend": "Blend",
    "lstm": "LSTM",
}

MODEL_COLOR = {
    "logreg": "#9ecae1",
    "svm_linear": "#9ecae1",
    "svm_rbf": "#6baed6",
    "random_forest": "#4292c6",
    "xgboost": "#2171b5",
    "lightgbm": "#084594",
    "blend": "#31a354",
    "lstm": "#f28e2b",
}


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _confusion_from_counts(cm: List[List[int]]):
    import numpy as np

    arr = np.array(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return arr, arr / row_sums


def _plot_confusion(cm: List[List[int]], title: str, out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    counts, norm = _confusion_from_counts(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")

    labels = ["Benign (0)", "Malicious (1)"]
    ax.set_xticks([0, 1], labels=labels, rotation=15, ha="right")
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label", labelpad=12)
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(counts[i, j])}\n({norm[i, j]:.2f})",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    # Reserve extra left margin so the y-axis label isn't clipped when exported.
    fig.tight_layout(rect=(0.10, 0.0, 1.0, 1.0))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def _ordered_models(models: Dict[str, Dict[str, object]]) -> List[str]:
    comparable = [m for m, payload in models.items() if "threshold_best_f1" in payload]
    present = [m for m in PREFERRED_MODEL_ORDER if m in comparable]
    extras = sorted([m for m in comparable if m not in set(PREFERRED_MODEL_ORDER)])
    return present + extras


def _plot_model_auprc(
    models: Dict[str, Dict[str, object]],
    lstm_auprc: float,
    out_path: str,
    blend_auprc: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ordered = _ordered_models(models)
    names = list(ordered)
    values = [float(models[n]["threshold_best_f1"]["auprc"]) for n in ordered]
    if blend_auprc is not None:
        names.append("blend")
        values.append(float(blend_auprc))
    names.append("lstm")
    values.append(float(lstm_auprc))
    labels = [DISPLAY_NAME.get(n, n) for n in names]
    colors = [MODEL_COLOR.get(n, "#4c4c4c") for n in names]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.bar(labels, values, color=colors)
    y_min = max(0.0, min(values) - 0.08)
    y_max = min(1.0, max(values) + 0.04)
    if y_max - y_min < 0.15:
        y_min = max(0.0, y_max - 0.15)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Model Comparison (Test Split)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.004,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", labelrotation=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _extract_curve_points(y_true: List[int], y_score: List[float]) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
    from sklearn.metrics import precision_recall_curve, roc_curve

    p, r, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return (r.tolist(), p.tolist()), (fpr.tolist(), tpr.tolist())


def _plot_curves(
    model_predictions: Dict[str, Dict[str, List[float]]],
    out_pr: str,
    out_roc: str,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, roc_auc_score

    palette = {
        "logreg": MODEL_COLOR["logreg"],
        "svm_linear": MODEL_COLOR["svm_linear"],
        "svm_rbf": MODEL_COLOR["svm_rbf"],
        "random_forest": MODEL_COLOR["random_forest"],
        "xgboost": MODEL_COLOR["xgboost"],
        "lightgbm": MODEL_COLOR["lightgbm"],
        "blend": MODEL_COLOR["blend"],
    }

    fig_pr, ax_pr = plt.subplots(figsize=(4.8, 4.1))
    fig_roc, ax_roc = plt.subplots(figsize=(4.8, 4.1))

    for name, pred in model_predictions.items():
        y_true = pred["y_true"]
        y_score = pred["y_score"]
        (recall, precision), (fpr, tpr) = _extract_curve_points(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        color = palette.get(name, "#4c4c4c")
        display = DISPLAY_NAME.get(name, name)
        ax_pr.plot(recall, precision, lw=1.8, color=color, label=f"{display} (AUPRC={ap:.3f})")
        ax_roc.plot(fpr, tpr, lw=1.8, color=color, label=f"{display} (AUC={auc:.3f})")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves (Test)")
    ax_pr.grid(alpha=0.25, linestyle="--")
    ax_pr.legend(fontsize=8, loc="lower left")
    fig_pr.tight_layout()
    fig_pr.savefig(out_pr, dpi=220)
    plt.close(fig_pr)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1.2)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves (Test)")
    ax_roc.grid(alpha=0.25, linestyle="--")
    ax_roc.legend(fontsize=8, loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(out_roc, dpi=220)
    plt.close(fig_roc)


def _plot_early_observation(early_payload: Dict[str, object], out_path: str) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(early_payload["results"], key=lambda r: float(r["fraction"]))
    x = [float(r["fraction_pct"]) for r in rows]
    auprc = [float(r["test"]["auprc"]) for r in rows]
    f1 = [float(r["test"]["f1"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(x, auprc, marker="o", lw=2.0, color="#1f77b4", label="AUPRC")
    ax.plot(x, f1, marker="s", lw=2.0, color="#d62728", label="F1")
    ax.set_xlabel("Observed stream fraction (%)")
    ax.set_ylabel("Test metric")
    ax.set_title("Early-Observation Attack Curve (XGBoost)")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_cv_stability(cv_payload: Dict[str, object], out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["logreg", "svm_rbf", "random_forest", "xgboost", "lightgbm"]
    labels = ["LogReg", "SVM-RBF", "RF", "XGBoost", "LightGBM"]
    reports = cv_payload.get("reports", {})

    auprc_data = []
    f1_data = []
    use_labels = []
    for key, label in zip(model_order, labels):
        if key not in reports:
            continue
        folds = reports[key]["folds"]
        auprc_data.append([float(f["auprc"]) for f in folds])
        f1_data.append([float(f["f1_at_0.5"]) for f in folds])
        use_labels.append(label)

    if not auprc_data:
        return

    positions = np.arange(1, len(use_labels) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    metric_specs = [
        (axes[0], auprc_data, "Fold AUPRC", "AUPRC"),
        (axes[1], f1_data, "Fold F1@0.5", "F1@0.5"),
    ]

    for ax, data, title, ylabel in metric_specs:
        vio = ax.violinplot(data, positions=positions, widths=0.85, showmeans=False, showextrema=False)
        for body in vio["bodies"]:
            body.set_facecolor("#7aa6d8")
            body.set_alpha(0.35)
            body.set_edgecolor("#3a6ea5")

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.28,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#1f1f1f", "linewidth": 1.3},
            boxprops={"facecolor": "#2c7fb8", "alpha": 0.65, "linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        # Keep lint-like tools happy for assigned local var in simple scripts.
        _ = bp

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions, labels=use_labels, rotation=22, ha="right")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Group-CV Stability Across Tabular Models", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_precision_at_k_curve(pred_payloads: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from analysis.common import precision_at_k

    # Top-k alerting operating points, where k is percent of traffic flagged.
    k_ratios = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    x = [k * 100 for k in k_ratios]

    palette = {
        "xgboost": MODEL_COLOR["xgboost"],
        "lightgbm": MODEL_COLOR["lightgbm"],
        "blend": MODEL_COLOR["blend"],
        "lstm": "#d95f0e",
    }
    labels = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "blend": "Blend", "lstm": "LSTM"}

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    plotted = 0
    for model_name in ["xgboost", "lightgbm", "blend", "lstm"]:
        if model_name not in pred_payloads:
            continue
        y_true = np.array(pred_payloads[model_name]["y_true"], dtype=int)
        y_score = np.array(pred_payloads[model_name]["y_score"], dtype=float)
        vals = [float(precision_at_k(y_true, y_score, k)) for k in k_ratios]
        ax.plot(
            x,
            vals,
            marker="o",
            lw=2.0,
            color=palette.get(model_name, "#4c4c4c"),
            label=labels.get(model_name, model_name),
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Flagged traffic (top % by model score)")
    ax.set_ylabel("Precision@k")
    ax.set_title("Operational Triage Curve (Top-k Precision)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(x), max(x))
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures from experiment artifacts")
    parser.add_argument("--tabular-metrics", default="artifacts/reports/tabular_metrics.json")
    parser.add_argument("--lstm-metrics", default="artifacts/reports/lstm_metrics.json")
    parser.add_argument("--pred-dir", default="artifacts/reports/predictions")
    parser.add_argument("--blend-metrics", default="artifacts/reports/blend_metrics.json")
    parser.add_argument("--early-metrics", default="artifacts/reports/early_observation_metrics.json")
    parser.add_argument("--group-cv", default="artifacts/reports/group_cv_metrics.json")
    parser.add_argument("--out-dir", default="artifacts/reports/figures")
    args = parser.parse_args()

    from analysis.common import ensure_dir

    ensure_dir(args.out_dir)

    tabular = _load_json(args.tabular_metrics)
    lstm_payload = _load_json(args.lstm_metrics)
    lstm_metrics = lstm_payload["lstm"]["threshold_best_f1"]
    blend_path = Path(args.blend_metrics)
    blend_payload = _load_json(str(blend_path)) if blend_path.exists() else {}
    blend_auprc = (
        float(blend_payload["best_blend"]["test_metrics"]["auprc"])
        if blend_payload.get("best_blend")
        else None
    )

    _plot_model_auprc(
        tabular["models"],
        lstm_auprc=float(lstm_metrics["auprc"]),
        blend_auprc=blend_auprc,
        out_path=f"{args.out_dir}/model_auprc_comparison.png",
    )

    tabular_models = tabular.get("models", {})
    comparable_models = {k: v for k, v in tabular_models.items() if "threshold_best_f1" in v}
    best_tabular_name = max(
        comparable_models,
        key=lambda n: float(comparable_models[n]["threshold_best_f1"]["auprc"]),
    )
    _plot_confusion(
        comparable_models[best_tabular_name]["threshold_best_f1"]["confusion_matrix"],
        f"{DISPLAY_NAME.get(best_tabular_name, best_tabular_name)} Confusion Matrix (Test)",
        f"{args.out_dir}/best_tabular_confusion_matrix.png",
    )
    if "xgboost" in tabular_models:
        _plot_confusion(
            tabular_models["xgboost"]["threshold_best_f1"]["confusion_matrix"],
            "XGBoost Confusion Matrix (Test)",
            f"{args.out_dir}/xgboost_confusion_matrix.png",
        )
    if blend_payload.get("best_blend"):
        _plot_confusion(
            blend_payload["best_blend"]["test_metrics"]["confusion_matrix"],
            "Blend Confusion Matrix (Test)",
            f"{args.out_dir}/blend_confusion_matrix.png",
        )
    _plot_confusion(
        lstm_metrics["confusion_matrix"],
        "LSTM Confusion Matrix (Test)",
        f"{args.out_dir}/lstm_confusion_matrix.png",
    )

    model_predictions: Dict[str, Dict[str, List[float]]] = {}
    for name in _ordered_models(tabular_models):
        pred_path = Path(args.pred_dir) / f"{name}_predictions.json"
        if not pred_path.exists():
            continue
        payload = _load_json(str(pred_path))["test"]
        model_predictions[name] = {
            "y_true": [int(x) for x in payload["y_true"]],
            "y_score": [float(x) for x in payload["y_score"]],
        }
    if blend_payload.get("best_blend"):
        weights = blend_payload["best_blend"]["weights"]
        if all(model in model_predictions for model in weights):
            import numpy as np

            y_true = model_predictions[next(iter(weights))]["y_true"]
            blend_scores = np.zeros(len(y_true), dtype=float)
            for model, weight in weights.items():
                blend_scores += float(weight) * np.array(model_predictions[model]["y_score"], dtype=float)
            model_predictions["blend"] = {
                "y_true": y_true,
                "y_score": blend_scores.tolist(),
            }

    if model_predictions:
        _plot_curves(
            model_predictions=model_predictions,
            out_pr=f"{args.out_dir}/tabular_pr_curves.png",
            out_roc=f"{args.out_dir}/tabular_roc_curves.png",
        )

    early_path = Path(args.early_metrics)
    if early_path.exists():
        early_payload = _load_json(str(early_path))
        _plot_early_observation(early_payload, f"{args.out_dir}/early_observation_curve.png")

    cv_path = Path(args.group_cv)
    if cv_path.exists():
        cv_payload = _load_json(str(cv_path))
        _plot_cv_stability(cv_payload, f"{args.out_dir}/cv_stability_violin.png")

    topk_models: Dict[str, Dict[str, List[float]]] = {}
    for name in ["xgboost", "lightgbm", "lstm"]:
        pred_path = Path(args.pred_dir) / f"{name}_predictions.json"
        if not pred_path.exists():
            continue
        payload = _load_json(str(pred_path))
        if "test" not in payload:
            continue
        topk_models[name] = {
            "y_true": [int(x) for x in payload["test"]["y_true"]],
            "y_score": [float(x) for x in payload["test"]["y_score"]],
        }
    if "blend" in model_predictions:
        topk_models["blend"] = model_predictions["blend"]
    if topk_models:
        _plot_precision_at_k_curve(topk_models, f"{args.out_dir}/precision_at_k_curve.png")

    print(f"Figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
