#!/usr/bin/env python3
"""Common utilities for encrypted-traffic intent analysis."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class DatasetRow:
    sample_id: str
    prompt_id: str
    label: int
    chatbot_name: str
    trial: int
    temperature: float
    timestamp: float
    response_token_count: int
    response_token_count_empty: int
    response_token_count_nonempty: int
    data_lengths: List[float]
    time_diffs: List[float]


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 1337


@dataclass
class SequenceConfig:
    max_len: int = 1024


@dataclass
class TrainingConfig:
    random_seed: int = 1337
    class_weight: str = "balanced"
    bootstrap_samples: int = 500
    precision_at_k: Tuple[float, ...] = (0.01, 0.05, 0.10)


@dataclass
class PathsConfig:
    data_glob: str = "data/shard*/GPT5MiniOpenRouter_shard*of50.json"
    prompts_path: str = "prompts/prompts.json"
    artifacts_dir: str = "artifacts"
    dataset_dir: str = "artifacts/datasets"
    models_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"


@dataclass
class ExperimentConfig:
    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)
    split: SplitConfig = dataclasses.field(default_factory=SplitConfig)
    sequence: SequenceConfig = dataclasses.field(default_factory=SequenceConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)


def maybe_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def load_config(path: str) -> ExperimentConfig:
    raw: Dict[str, object]
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        yaml_mod = maybe_import_yaml()
        if yaml_mod is None:
            raise RuntimeError(
                "PyYAML is required to read YAML config. Install with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml_mod.safe_load(f)

    paths = PathsConfig(**raw.get("paths", {}))
    split = SplitConfig(**raw.get("split", {}))
    sequence = SequenceConfig(**raw.get("sequence", {}))
    training = TrainingConfig(**raw.get("training", {}))
    return ExperimentConfig(paths=paths, split=split, sequence=sequence, training=training)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def discover_shard_files(glob_pattern: str) -> List[str]:
    import glob

    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No shard json files found for pattern: {glob_pattern}")
    return files


def load_prompt_sets(prompts_path: str) -> Tuple[set, set]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    pos = set(prompts["positive"]["prompts"])
    neg = set(prompts["negative"]["prompts"])
    return pos, neg


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_labeled_rows(config: ExperimentConfig) -> Tuple[List[DatasetRow], Dict[str, int]]:
    files = discover_shard_files(config.paths.data_glob)
    positive_prompts, negative_prompts = load_prompt_sets(config.paths.prompts_path)

    rows: List[DatasetRow] = []
    stats = {
        "files": len(files),
        "rows_total": 0,
        "rows_kept": 0,
        "rows_unmatched_prompt": 0,
        "rows_bad_seq": 0,
    }

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue

        for item in data:
            stats["rows_total"] += 1

            prompt = item.get("prompt")
            if prompt in positive_prompts:
                label = 1
            elif prompt in negative_prompts:
                label = 0
            else:
                stats["rows_unmatched_prompt"] += 1
                continue

            data_lengths = item.get("data_lengths") or []
            time_diffs = item.get("time_diffs") or []
            if not isinstance(data_lengths, list) or not isinstance(time_diffs, list):
                stats["rows_bad_seq"] += 1
                continue
            if len(data_lengths) != len(time_diffs) or len(data_lengths) == 0:
                stats["rows_bad_seq"] += 1
                continue

            trial = _safe_int(item.get("trial"), 0)
            timestamp = _safe_float(item.get("timestamp"), 0.0)
            prompt_id = sha1_text(prompt)
            sample_id = sha1_text(f"{prompt}|{trial}|{timestamp}")

            rows.append(
                DatasetRow(
                    sample_id=sample_id,
                    prompt_id=prompt_id,
                    label=label,
                    chatbot_name=str(item.get("chatbot_name", "unknown")),
                    trial=trial,
                    temperature=_safe_float(item.get("temperature"), 0.0),
                    timestamp=timestamp,
                    response_token_count=_safe_int(item.get("response_token_count"), 0),
                    response_token_count_empty=_safe_int(item.get("response_token_count_empty"), 0),
                    response_token_count_nonempty=_safe_int(item.get("response_token_count_nonempty"), 0),
                    data_lengths=[_safe_float(x) for x in data_lengths],
                    time_diffs=[_safe_float(x) for x in time_diffs],
                )
            )
            stats["rows_kept"] += 1

    return rows, stats


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / len(values)))


def _quantiles(values: Sequence[float], qs: Sequence[float]) -> Dict[float, float]:
    if not values:
        return {q: 0.0 for q in qs}
    sorted_vals = sorted(values)
    out = {}
    n = len(sorted_vals)
    for q in qs:
        idx = min(max(int(round((n - 1) * q)), 0), n - 1)
        out[q] = float(sorted_vals[idx])
    return out


def _entropy(values: Sequence[float], bins: int = 20) -> float:
    if not values:
        return 0.0
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        return 0.0
    width = (vmax - vmin) / bins
    counts = [0] * bins
    for x in values:
        idx = int((x - vmin) / width)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1
    total = sum(counts)
    ent = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def _run_length_stats(values: Sequence[float], threshold: Optional[float] = None) -> Tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    if threshold is None:
        threshold = _mean(values)
    runs: List[int] = []
    current = 0
    for v in values:
        if v >= threshold:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    if not runs:
        return (0.0, 0.0, 0.0)

    return (float(len(runs)), _mean(runs), float(max(runs)))


def _topk_desc(values: Sequence[float], k: int = 5) -> List[float]:
    sorted_vals = sorted(values, reverse=True)
    out = sorted_vals[:k]
    if len(out) < k:
        out += [0.0] * (k - len(out))
    return [float(x) for x in out]


def row_to_feature_dict(row: DatasetRow) -> Dict[str, float]:
    size = row.data_lengths
    time = row.time_diffs
    token_total = float(row.response_token_count)
    token_empty = float(row.response_token_count_empty)
    token_empty_pct = (token_empty / token_total) if token_total > 0.0 else 0.0

    size_q = _quantiles(size, [0.1, 0.25, 0.5, 0.75, 0.9])
    time_q = _quantiles(time, [0.1, 0.25, 0.5, 0.75, 0.9])
    size_runs = _run_length_stats(size)
    time_runs = _run_length_stats(time)
    size_top = _topk_desc(size, 5)

    features: Dict[str, float] = {
        "packet_count": float(len(size)),
        "size_mean": _mean(size),
        "size_std": _std(size),
        "size_min": float(min(size) if size else 0.0),
        "size_max": float(max(size) if size else 0.0),
        "time_mean": _mean(time),
        "time_std": _std(time),
        "time_min": float(min(time) if time else 0.0),
        "time_max": float(max(time) if time else 0.0),
        "size_entropy": _entropy(size),
        "time_entropy": _entropy(time),
        "size_run_count": size_runs[0],
        "size_run_mean": size_runs[1],
        "size_run_max": size_runs[2],
        "time_run_count": time_runs[0],
        "time_run_mean": time_runs[1],
        "time_run_max": time_runs[2],
        "response_token_count_empty_pct": token_empty_pct,
        "temperature": float(row.temperature),
        "trial": float(row.trial),
    }

    for q, v in size_q.items():
        features[f"size_q{int(q*100)}"] = v
    for q, v in time_q.items():
        features[f"time_q{int(q*100)}"] = v
    for i, v in enumerate(size_top, start=1):
        features[f"size_top{i}"] = v

    return features


def group_stratified_split(
    prompt_ids: Sequence[str], labels: Sequence[int], split_config: SplitConfig
) -> Dict[str, set]:
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
    except Exception as ex:
        raise RuntimeError(
            "scikit-learn is required for group stratified split. Install with: pip install scikit-learn"
        ) from ex

    group_to_label: Dict[str, int] = {}
    for gid, label in zip(prompt_ids, labels):
        if gid not in group_to_label:
            group_to_label[gid] = int(label)

    groups = list(group_to_label.keys())
    group_labels = [group_to_label[g] for g in groups]

    train_groups, rem_groups, train_labels, rem_labels = train_test_split(
        groups,
        group_labels,
        test_size=(1.0 - split_config.train_ratio),
        random_state=split_config.random_seed,
        stratify=group_labels,
    )

    rem_total = split_config.val_ratio + split_config.test_ratio
    if rem_total <= 0:
        raise ValueError("val_ratio + test_ratio must be > 0")

    val_share_of_rem = split_config.val_ratio / rem_total
    val_groups, test_groups = train_test_split(
        rem_groups,
        test_size=(1.0 - val_share_of_rem),
        random_state=split_config.random_seed,
        stratify=rem_labels,
    )

    return {
        "train": set(train_groups),
        "val": set(val_groups),
        "test": set(test_groups),
    }


def assign_split(prompt_id: str, split_groups: Dict[str, set]) -> str:
    for split_name, groups in split_groups.items():
        if prompt_id in groups:
            return split_name
    raise KeyError(f"Prompt group not assigned to split: {prompt_id}")


def leak_checks(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    prompt_by_split: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    seq_by_split: Dict[str, set] = {"train": set(), "val": set(), "test": set()}

    for r in records:
        split = str(r["split"])
        prompt_id = str(r["prompt_id"])
        seq_hash = str(r["sequence_hash"])
        prompt_by_split[split].add(prompt_id)
        seq_by_split[split].add(seq_hash)

    def pair_overlap(a: set, b: set) -> int:
        return len(a.intersection(b))

    result = {
        "prompt_overlap": {
            "train_val": pair_overlap(prompt_by_split["train"], prompt_by_split["val"]),
            "train_test": pair_overlap(prompt_by_split["train"], prompt_by_split["test"]),
            "val_test": pair_overlap(prompt_by_split["val"], prompt_by_split["test"]),
        },
        "sequence_overlap": {
            "train_val": pair_overlap(seq_by_split["train"], seq_by_split["val"]),
            "train_test": pair_overlap(seq_by_split["train"], seq_by_split["test"]),
            "val_test": pair_overlap(seq_by_split["val"], seq_by_split["test"]),
        },
    }
    return result


def make_sequence_hash(data_lengths: Sequence[float], time_diffs: Sequence[float]) -> str:
    payload = json.dumps({"data_lengths": data_lengths, "time_diffs": time_diffs}, sort_keys=True)
    return sha1_text(payload)


def save_json(path: str, payload: object) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def to_parquet_or_csv(df, base_path_no_ext: str) -> str:
    """Save a dataframe to parquet if pyarrow is available, else csv."""
    out_parquet = f"{base_path_no_ext}.parquet"
    out_csv = f"{base_path_no_ext}.csv"
    try:
        df.to_parquet(out_parquet, index=False)
        return out_parquet
    except Exception:
        df.to_csv(out_csv, index=False)
        return out_csv


def load_tabular_table(path: str):
    try:
        import pandas as pd  # type: ignore
    except Exception as ex:
        raise RuntimeError("pandas is required. Install with: pip install pandas") from ex

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def feature_columns_from_df(df) -> List[str]:
    excluded = {
        "sample_id",
        "prompt_id",
        "label",
        "split",
        "sequence_hash",
        "chatbot_name",
    }
    return [c for c in df.columns if c not in excluded]


def print_env_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    package_name_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "torch": "torch",
        "pyyaml": "PyYAML",
    }
    for key, dist_name in package_name_map.items():
        try:
            versions[key] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "not-installed"
        except Exception:
            versions[key] = "unknown"
    return versions


def precision_at_k(y_true, y_score, k_ratio: float) -> float:
    import numpy as np  # type: ignore

    if len(y_true) == 0:
        return 0.0
    k = max(1, int(round(len(y_true) * k_ratio)))
    idx = np.argsort(-y_score)[:k]
    return float(np.mean(np.array(y_true)[idx]))


def bootstrap_metric_ci(y_true, y_score, metric_fn, n_bootstrap: int, seed: int = 1337) -> Tuple[float, float]:
    import numpy as np  # type: ignore

    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(np.array(y_true)[idx], np.array(y_score)[idx]))
    vals = np.array(vals)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def find_threshold_for_target_precision(y_true, y_score, target_precision: float = 0.90) -> float:
    import numpy as np  # type: ignore
    from sklearn.metrics import precision_recall_curve  # type: ignore

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    best = 0.5
    for p, t in zip(precision[:-1], thresholds):
        if p >= target_precision:
            best = float(t)
            break
    return best


def find_threshold_for_best_f1(y_true, y_score) -> float:
    import numpy as np  # type: ignore
    from sklearn.metrics import precision_recall_curve  # type: ignore

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx])
