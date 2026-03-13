#!/usr/bin/env python3
"""Qualitative error analysis from saved prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate error-analysis summaries from predictions.")
    parser.add_argument("--dataset", default="artifacts/datasets/openrouter_intent_features.csv")
    parser.add_argument("--predictions", required=True, help="Path to model predictions JSON.")
    parser.add_argument("--metrics", default="artifacts/reports/tabular_metrics.json")
    parser.add_argument("--model", required=True, help="Model name key inside metrics JSON.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    import pandas as pd

    from analysis.common import ensure_dir, load_tabular_table, save_json

    df = load_tabular_table(args.dataset)
    test_df = df[df["split"] == "test"].copy()

    with open(args.predictions, "r", encoding="utf-8") as f:
        pred_payload = json.load(f)
    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics_payload = json.load(f)

    threshold = float(metrics_payload["models"][args.model]["threshold_best_f1"]["threshold"])
    pred = pred_payload["test"]
    pred_df = pd.DataFrame(
        {
            "sample_id": pred["sample_id"],
            "prompt_id": pred["prompt_id"],
            "y_true": pred["y_true"],
            "y_score": pred["y_score"],
        }
    )
    pred_df["y_pred"] = (pred_df["y_score"] >= threshold).astype(int)
    pred_df["error_type"] = "correct"
    pred_df.loc[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1), "error_type"] = "FP"
    pred_df.loc[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0), "error_type"] = "FN"

    merged = pred_df.merge(
        test_df[
            [
                "sample_id",
                "packet_count",
                "time_std",
                "size_std",
                "time_entropy",
                "size_entropy",
            ]
        ],
        on="sample_id",
        how="left",
    )

    feature_cols = ["packet_count", "time_std", "size_std", "time_entropy", "size_entropy"]
    buckets: Dict[str, object] = {}
    for key, subset in {
        "TP": merged[(merged["y_true"] == 1) & (merged["y_pred"] == 1)],
        "TN": merged[(merged["y_true"] == 0) & (merged["y_pred"] == 0)],
        "FP": merged[merged["error_type"] == "FP"],
        "FN": merged[merged["error_type"] == "FN"],
    }.items():
        out = {
            "count": int(len(subset)),
            "feature_means": {c: float(subset[c].mean()) if len(subset) else 0.0 for c in feature_cols},
        }
        buckets[key] = out

    mistakes = merged[merged["error_type"].isin(["FP", "FN"])].copy()
    mistakes["margin_from_threshold"] = (mistakes["y_score"] - threshold).abs()
    mistakes = mistakes.sort_values("margin_from_threshold", ascending=False)
    top = mistakes.head(args.top_k).copy()

    summary = {
        "model": args.model,
        "threshold_best_f1": threshold,
        "counts": {
            "test_total": int(len(merged)),
            "errors_total": int((merged["error_type"] != "correct").sum()),
            "fp": int((merged["error_type"] == "FP").sum()),
            "fn": int((merged["error_type"] == "FN").sum()),
        },
        "buckets": buckets,
        "top_confident_errors": top[
            [
                "sample_id",
                "prompt_id",
                "y_true",
                "y_score",
                "y_pred",
                "error_type",
                "margin_from_threshold",
            ]
            + feature_cols
        ].to_dict(orient="records"),
    }

    out_json = args.output_json or f"artifacts/reports/{args.model}_error_analysis.json"
    out_csv = args.output_csv or f"artifacts/reports/{args.model}_top_errors.csv"
    ensure_dir(str(Path(out_json).parent))
    ensure_dir(str(Path(out_csv).parent))

    save_json(out_json, summary)
    top.to_csv(out_csv, index=False)
    print(f"Error analysis JSON: {out_json}")
    print(f"Top errors CSV: {out_csv}")


if __name__ == "__main__":
    main()
