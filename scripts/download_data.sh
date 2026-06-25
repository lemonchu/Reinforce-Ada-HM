#!/bin/bash
# =============================================================================
# download_data.sh — "Plan B" data prep: download a pre-processed RLHFlow prompt
# set from HuggingFace and reformat it into verl parquet, instead of generating
# from scratch (no 8-GPU job needed). Runs fine on the login node (CPU-only).
#
# Output: ${LOCAL_DIR}/{train,test}.parquet. By default LOCAL_DIR is the repo's
# ./data/openr1 symlinked onto Ocean (home is nearly full), so the training
# scripts that hardcode ./data/openr1 still work.
#
# Usage:
#   bash scripts/download_data.sh            # hard prompts (default)
#   LEVEL=easy bash scripts/download_data.sh # easy prompts
# =============================================================================

set -xeuo pipefail

# ----------------------------------------------------------------------------
# Configurable knobs
# ----------------------------------------------------------------------------
LEVEL="${LEVEL:-hard}"                       # hard | easy
# Persistent data home on Ocean (cis250185p has the most free space).
DATA_DIR="${DATA_DIR:-/ocean/projects/cis250185p/tming/reinforce-ada/data/openr1}"
# Model only used by reformat.py for a token-length filter; README default.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Math-1.5B}"
# Keep HuggingFace's download cache off the small home filesystem.
export HF_HOME="${HF_HOME:-/ocean/projects/cis250185p/tming/hf_cache}"

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Pick the HF dataset for the requested difficulty level (1.5B prompt sets).
# NOTE: the README link TEXT (e.g. ..._hard_prompt_1.5b) does NOT match the real
# HF repo id — use the actual ids from the README hyperlink URLs: the 1.5B sets
# use a dash ("1-5b"), and the easy set is named "simple".
case "${LEVEL}" in
    hard) DATA_SOURCE="RLHFlow/reinforce_ada_hard_prompt_1-5b" ;;
    easy) DATA_SOURCE="RLHFlow/reinforce_ada_simple_prompt_1-5b" ;;
    *) echo "ERROR: LEVEL must be 'hard' or 'easy', got '${LEVEL}'." >&2; exit 1 ;;
esac

cd "${REPO_DIR}"

# ----------------------------------------------------------------------------
# Step 1 — Create the Ocean data dir and link ./data/openr1 to it.
# ----------------------------------------------------------------------------
mkdir -p "${DATA_DIR}" data
ln -sfn "${DATA_DIR}" data/openr1            # ./data/openr1 -> Ocean
LOCAL_DIR="./data/openr1"

# ----------------------------------------------------------------------------
# Step 2 — Download + reformat the prompt set into train.parquet.
# ----------------------------------------------------------------------------
python3 -m data_process.reformat \
    --local_dir "${LOCAL_DIR}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_source "${DATA_SOURCE}"

# ----------------------------------------------------------------------------
# Step 3 — Build the validation set -> test.parquet.
# ----------------------------------------------------------------------------
python3 -m data_process.get_validation_set \
    --local_dir "${LOCAL_DIR}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}"

# ----------------------------------------------------------------------------
# Step 4 — Verify.
# ----------------------------------------------------------------------------
ls -la "${LOCAL_DIR}/train.parquet" "${LOCAL_DIR}/test.parquet"
echo "Done. Data (${LEVEL} prompts) ready at ${DATA_DIR} (linked as ${LOCAL_DIR})."
