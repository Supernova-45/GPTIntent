# GPTIntent

Repository layout has been organized by purpose:

- `analysis/`: ML experiments for encrypted-traffic intent inference
- `src/collection/`: dataset collection scripts
- `prompts/`: prompt-generation code and prompt datasets
- `scripts/`: operational scripts
- `env/`: environment specs
- `reproducibility/`: exported lock files and package manifests
- `data/`: collected shard data
- `artifacts/`: model/dataset/report outputs

## Quick start

### 1) Create environment

```bash
bash scripts/create_conda_env.sh
```

This installs the full analysis runtime needed by default (`xgboost`, `lightgbm`, `torch`) on top of the base conda environment.

### 2) Run analysis pipeline

```bash
python3 analysis/01_build_dataset.py --config analysis/config.yaml
python3 analysis/02_train_tabular.py --config analysis/config.yaml
python3 analysis/03_train_lstm.py --config analysis/config.yaml
python3 analysis/04_ablations.py --config analysis/config.yaml
```

### 3) Collection scripts

- Parallel setup helper: `scripts/parallel.sh`
- Fast collector: `src/collection/collect_fast.py`
- OpenRouter chatbot shim: `src/collection/gpt_5_mini_openrouter.py`

Detailed analysis instructions are in `analysis/README.md`.

## Dataset features (`openrouter_intent_features.csv`)

The dataset is built from encrypted traffic traces collected during chatbot API calls. **Packet sizes** are the byte length of each network packet (request/response chunks). **Inter-packet timings** are the elapsed time (seconds) between consecutive packets. Both are observable on encrypted links and can leak information about application behavior. Each row is one sample; the label (0/1) indicates prompt intent (negative/positive). Features are derived from `data_lengths` (packet sizes in bytes) and `time_diffs` (inter-packet intervals in seconds).

| Feature | Description |
|---------|-------------|
| `packet_count` | Number of packets in the trace |
| `size_mean`, `size_std`, `size_min`, `size_max` | Mean, std, min, max of packet sizes (bytes) |
| `size_entropy` | Shannon entropy of packet size distribution (20 bins) |
| `size_run_count`, `size_run_mean`, `size_run_max` | Count, mean length, and max length of runs of consecutive packets above mean size |
| `size_q10`, `size_q25`, `size_q50`, `size_q75`, `size_q90` | 10th–90th percentiles of packet sizes |
| `size_top1` … `size_top5` | Five largest packet sizes |
| `time_mean`, `time_std`, `time_min`, `time_max` | Mean, std, min, max of inter-packet intervals (seconds) |
| `time_entropy` | Shannon entropy of inter-packet timing distribution |
| `time_run_count`, `time_run_mean`, `time_run_max` | Run-length stats for intervals above mean |
| `time_q10`, `time_q25`, `time_q50`, `time_q75`, `time_q90` | 10th–90th percentiles of inter-packet intervals |
| `response_token_count_empty_pct` | Fraction of response tokens that were empty |
| `temperature` | Model sampling temperature used for generation |

Metadata columns (not used as model features): `sample_id`, `prompt_id`, `label`, `split`, `sequence_hash`, `chatbot_name`, `trial`.
