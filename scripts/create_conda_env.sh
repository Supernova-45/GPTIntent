#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="env/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing ${ENV_FILE}"
  exit 1
fi

ENV_NAME="$(awk '/^name:/ {print $2; exit}' "${ENV_FILE}")"
if [ -z "${ENV_NAME}" ]; then
  echo "Could not parse environment name from ${ENV_FILE}"
  exit 1
fi

conda env create -f "${ENV_FILE}" || conda env update -f "${ENV_FILE}" --prune

echo "Installing analysis extras (xgboost, lightgbm, pytorch, numpy<2)..."
conda install -n "${ENV_NAME}" -y -c conda-forge xgboost lightgbm "numpy<2" pytorch

mkdir -p reproducibility

# Portable environment spec
conda env export -n "${ENV_NAME}" --no-builds > reproducibility/environment_export_no_builds.yml

# Platform-locked explicit list
conda list -n "${ENV_NAME}" --explicit > reproducibility/conda_explicit_lock.txt

# Pip packages in the environment
conda run -n "${ENV_NAME}" python -m pip freeze > reproducibility/pip_freeze.txt

echo "Environment created/updated: ${ENV_NAME}"
echo "Wrote reproducibility/environment_export_no_builds.yml"
echo "Wrote reproducibility/conda_explicit_lock.txt"
echo "Wrote reproducibility/pip_freeze.txt"
