"""Build labeled feature table from shard JSONs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.common import (
    assign_split,
    ensure_dir,
    group_stratified_split,
    leak_checks,
    load_config,
    load_labeled_rows,
    make_sequence_hash,
    print_env_versions,
    row_to_feature_dict,
    save_json,
    set_seed,
    to_parquet_or_csv,
)


def main() -> None:
    config_path = str(Path(__file__).resolve().with_name("config.yaml"))

    config = load_config(config_path)
    set_seed(config.training.random_seed)

    ensure_dir(config.paths.dataset_dir)
    ensure_dir(config.paths.reports_dir)

    rows, load_stats = load_labeled_rows(config)

    prompt_ids = [r.prompt_id for r in rows]
    labels = [r.label for r in rows]
    split_groups = group_stratified_split(prompt_ids, labels, config.split)

    records = []
    for r in rows:
        feat = row_to_feature_dict(r)
        rec = {
            "sample_id": r.sample_id,
            "prompt_id": r.prompt_id,
            "label": r.label,
            "split": assign_split(r.prompt_id, split_groups),
            "sequence_hash": make_sequence_hash(r.data_lengths, r.time_diffs),
            "chatbot_name": r.chatbot_name,
        }
        rec.update(feat)
        records.append(rec)

    df = pd.DataFrame(records)

    base = f"{config.paths.dataset_dir}/openrouter_intent_features"
    dataset_path = to_parquet_or_csv(df, base)

    split_counts = Counter(df["split"].tolist())
    label_counts = Counter(df["label"].tolist())

    split_label_counts = {}
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        split_label_counts[split_name] = {
            "total": int(len(sub)),
            "label_0": int((sub["label"] == 0).sum()),
            "label_1": int((sub["label"] == 1).sum()),
        }

    leakage = leak_checks(records)

    qc = {
        "dataset_path": dataset_path,
        "load_stats": load_stats,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "split_label_counts": split_label_counts,
        "leakage": leakage,
        "env_versions": print_env_versions(),
    }

    qc_path = f"{config.paths.reports_dir}/dataset_qc.json"
    save_json(qc_path, qc)

    if leakage["prompt_overlap"]["train_val"] != 0 or leakage["prompt_overlap"]["train_test"] != 0 or leakage["prompt_overlap"]["val_test"] != 0:
        raise ValueError("Prompt leakage detected across splits.")

    print(f"Built dataset: {dataset_path}")
    print(f"QC report: {qc_path}")
    print("Split counts:", dict(split_counts))
    print("Label counts:", dict(label_counts))


if __name__ == "__main__":
    main()
