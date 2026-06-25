#!/bin/bash
# =============================================================================
# build_env.sh — Build the Reinforce-Ada-HM Python environment (PSC Bridges-2)
#
# Package manager: uv (ALWAYS). Target stack:
#   Python 3.10 | torch 2.7.1+cu126 | flash-attn 2.8.0.post2 | vllm 0.10.1
#   (README says torch 2.6.0, but vllm 0.10.1 pulls 2.7.1 — see Step 5 note.)
#
# -----------------------------------------------------------------------------
# WORKFLOW: build the env ONCE on a cheap, light GPU node, then SUBMIT training
# as a separate sbatch job that activates this same venv. You do NOT need a big
# (H100 / multi-GPU) node just to build — building only needs nvcc + one GPU to
# compile flash-attn and run the import sanity check. Save the expensive GPUs
# for the actual training job.
#
# Do NOT run this on the login node: flash-attn compiles from source
# (--no-build-isolation) and needs nvcc + a GPU; the login node has neither
# (only system Python 3.6.8, no CUDA). The script guards against this.
#
# STEP 0 — Grab a LIGHT interactive H100 node, then run this script there:
#     (V100 cannot be used — TORCH_CUDA_ARCH_LIST=9.0 targets sm_90/H100 only)
#
#     interact -p GPU-shared --gres=gpu:h100-80:1 -t 08:00:00
#
#     # Once on the compute node (shell hostname changes):
#     cd /jet/home/tming/Reinforce-Ada-HM
#     bash scripts/build_env.sh            # run all steps
#     bash scripts/build_env.sh 6          # RESUME from step 6 (e.g. flash-attn)
#     START_STEP=7 bash scripts/build_env.sh   # same, via env var
#
#   The optional arg is the step to START FROM (default 1). It only gates the
#   install steps (5=torch, 6=flash-attn, 7=deps/-e/vllm, 8=sanity check); the
#   prerequisite steps (1=guard, 2=cuda module, 3=uv PATH, 4=venv activate)
#   ALWAYS run so the target venv is active before any install happens.
#
# STEP LATER — Run training as a batch job (NOT in this interactive session).
#   The venv lives at $VENV_DIR and is reusable; your sbatch script should
#   `module load cuda/12.4.0` and `source $VENV_DIR/bin/activate` before
#   launching, e.g. bash scripts/run_reinforce_ada.sh (defaults to 4 GPUs).
# =============================================================================

set -xeuo pipefail

# ----------------------------------------------------------------------------
# Configurable knobs
# ----------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-$HOME/.venvs/reinforce_ada}"   # where the venv lives
PY_VERSION="${PY_VERSION:-3.10}"                      # uv will fetch this if missing
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.0}"             # matches torch cu124 build
GCC_MODULE="${GCC_MODULE:-gcc/10.2.0}"                # cluster default gcc 8.5 is
                                                      # too old for torch 2.6 CUDA
                                                      # ext builds (need gcc >= 9)
# GPU archs to compile flash-attn for. Pinning this to ONLY your training GPU
# dramatically cuts build time (the default builds several archs).
#   H100 -> "9.0"   L40S -> "8.9"   A100 -> "8.0"   both H100+L40S -> "8.9 9.0"
# NOTE: FlashAttention-2 supports Ampere/Ada/Hopper only — NOT V100 (sm_70). Build
# and train on H100/L40S (this cluster has h100-80 and l40s-48 GPU nodes).
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # default: H100 only
# Torch is pinned EVERYWHERE it is mentioned so no install ever upgrades it (e.g.
# an unpinned `uv pip install flash-attn` would otherwise pull the latest torch).
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"               # the version vllm 0.10.1 needs
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Step to START FROM (default 1). CLI arg ($1) overrides the START_STEP env var.
# Only gates install steps 5-8; prerequisite steps 1-4 always run (see header).
START_STEP="${1:-${START_STEP:-1}}"
if ! [[ "${START_STEP}" =~ ^[1-8]$ ]]; then
    echo "ERROR: START_STEP must be an integer 1-8, got '${START_STEP}'." >&2
    exit 1
fi
# True when the given step number should run given START_STEP.
should_run() { [ "${START_STEP}" -le "$1" ]; }

# ----------------------------------------------------------------------------
# Step 1 — Guard: make sure we are NOT on the login node (need a GPU).
# ----------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — you appear to be on the login node."
    echo "       Grab a GPU node first (see STEP 0 in this script's header)."
    exit 1
fi
nvidia-smi

# ----------------------------------------------------------------------------
# Step 2 — Load CUDA (nvcc) and a modern GCC. The cluster default gcc is 8.5,
#          which fails to compile torch 2.6 CUDA extensions ("overflow in
#          constant expression" in DispatchKeySet.h); gcc >= 9 fixes it.
#          Export CC/CXX so torch's ninja build uses the loaded compiler.
# ----------------------------------------------------------------------------
module load "${CUDA_MODULE}"
module load "${GCC_MODULE}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
nvcc --version
gcc --version | head -1

# ----------------------------------------------------------------------------
# Step 3 — Make sure uv is on PATH (installed at ~/.local/bin/uv on this cluster).
# ----------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ----------------------------------------------------------------------------
# Step 4 — Create (or reuse) a Python 3.10 venv with uv, then activate it.
#          uv downloads Python 3.10 automatically if it is not already present.
# ----------------------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
    uv venv "${VENV_DIR}" --python "${PY_VERSION}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
uv pip install --upgrade pip

# ----------------------------------------------------------------------------
# Step 5 — Install torch FIRST. NOTE: the README says torch 2.6.0+cu124, but
#          vllm==0.10.1 (installed in Step 6) REQUIRES torch 2.7.1 and will
#          silently upgrade it. If flash-attn is built against 2.6.0 and torch is
#          then bumped to 2.7.1, flash-attn fails to import with an ABI error
#          ("undefined symbol: _ZN3c105Error..."). So we pin torch to the version
#          vllm wants (2.7.1+cu126) up front and build flash-attn LAST (Step 7)
#          against this same torch.
# ----------------------------------------------------------------------------
if should_run 5; then
    uv pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
fi

# ----------------------------------------------------------------------------
# Step 6 — Remaining deps, the package itself (editable), then vllm. Done BEFORE
#          flash-attn so the final torch version is settled first (vllm==0.10.1
#          must see torch 2.7.1 already satisfied -> no surprise upgrade).
#          requirements.txt pins numpy<2.0.0, ray[default], math_verify, etc.
# ----------------------------------------------------------------------------
if should_run 6; then
    cd "${REPO_DIR}"
    uv pip install -r requirements.txt
    uv pip install -e .
    uv pip install vllm==0.10.1
fi

# ----------------------------------------------------------------------------
# Step 7 — flash-attn (compiles from source). Built LAST, against the now-final
#          torch, so its compiled ABI matches the torch that will be loaded.
#          --no-build-isolation means the build needs its build tools
#          (setuptools/wheel/packaging/ninja) + torch ALREADY in the venv; numpy
#          and most of these arrived via Step 6, but install the build backend
#          explicitly to be safe.
#
#          NINJA — YES, this build uses ninja, and it is essential. Per the
#          flash-attention docs (github.com/Dao-AILab/flash-attention):
#            * "Without ninja, compiling can take a very long time (2h) since it
#               does not use multiple CPU cores. With ninja compiling takes 3-5
#               minutes on a 64-core machine."
#          torch's cpp_extension auto-detects ninja when it is installed — no flag
#          needed (you'll see "Emitting ninja build file" + "ninja -v -j N" in the
#          log). So we just make sure ninja is present and actually works.
#            * The docs warn ninja's health can be flaky: verify with
#              `ninja --version && echo $?` (must be 0); if not, the fix is
#              `pip uninstall -y ninja && pip install ninja`. We assert this below
#              so a broken ninja fails fast instead of silently dropping to the 2h
#              single-core path.
#            * MAX_JOBS caps ninja's parallel compile jobs. The docs note: with
#              <96GB RAM and many cores, ninja may spawn too many jobs and exhaust
#              RAM. Each flash-attn nvcc job needs ~3-5GB RAM, so a rough cap is
#              min(nproc, free_GB/5). We default to 16; lower it (e.g. MAX_JOBS=4)
#              on a small GPU-shared slice if it OOMs.
#
#          FLASH_ATTENTION_FORCE_BUILD skips flash-attn's attempt to download a
#          prebuilt wheel (which fails behind the cluster's TLS interception) and
#          compiles from source directly.
# ----------------------------------------------------------------------------
if should_run 7; then
    uv pip install "numpy<2.0.0" setuptools wheel packaging ninja
    # Assert ninja is healthy — otherwise the build silently falls back to the
    # ~2h single-core path (see docs note above).
    if ! ninja --version >/dev/null 2>&1; then
        echo "ninja is broken; reinstalling..." >&2
        uv pip install --force-reinstall ninja
    fi
    # TORCH_CUDA_ARCH_LIST limits which GPU archs are built (big build-time win);
    # see the knob near the top.
    #
    # Force a CLEAN rebuild of flash-attn ONLY, while keeping torch pinned:
    #   --reinstall-package flash-attn : rebuild just flash-attn (NOT --reinstall,
    #       which re-resolves the whole tree and would UPGRADE torch to latest).
    #   --no-cache : don't reuse a cached wheel (it's keyed by version, not torch,
    #       so a stale wheel built against another torch would cause an ABI error).
    #   torch==${TORCH_VERSION} listed explicitly : pins torch so the resolution
    #       can't bump it while satisfying flash-attn's unpinned `torch` requirement.
    export TORCH_CUDA_ARCH_LIST
    echo "Building flash-attn: archs=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS:-16}"
    MAX_JOBS="${MAX_JOBS:-16}" FLASH_ATTENTION_FORCE_BUILD=TRUE \
        uv pip install flash-attn==2.8.0.post2 "torch==${TORCH_VERSION}" \
            --no-build-isolation --reinstall-package flash-attn --no-cache
fi

# ----------------------------------------------------------------------------
# Step 8 — Sanity check: import the key libraries and confirm CUDA is visible.
# ----------------------------------------------------------------------------
if should_run 8; then
    python - <<'PY'
import torch, flash_attn, vllm, verl
print("torch     :", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("flash_attn:", flash_attn.__version__)
print("vllm      :", vllm.__version__)
print("verl imported OK")
PY
fi

echo
echo "=============================================================="
echo " Environment ready at: ${VENV_DIR}"
echo " Reuse it in any job/session with: source ${VENV_DIR}/bin/activate"
echo " Next: 'wandb login', then 'bash scripts/prepare_data.sh', then submit"
echo "       training as an sbatch job (module load ${CUDA_MODULE} + activate venv)."
echo "=============================================================="
