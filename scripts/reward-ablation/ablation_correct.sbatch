#!/bin/bash
# Reward ablation 1/3 — reward = correctness only.
#SBATCH --job-name=rwd-correct
#SBATCH --account=cis260009p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-80:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=/ocean/projects/cis250185p/tming/reinforce-ada/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis250185p/tming/reinforce-ada/logs/%x-%j.err

set -xeuo pipefail

# --- Environment (Bridges-2): CUDA + modern GCC + the prebuilt uv venv ---------
module load cuda/12.4.0 gcc/10.2.0
source /jet/home/tming/.venvs/reinforce_ada/bin/activate

cd /jet/home/tming/Reinforce-Ada-HM

# --- Reward composition for this run -------------------------------------------
export REWARD_COMPONENTS="[correctness]"
export EXP_TAG="correct"

bash scripts/reward-ablation/_train_reward_ablation.sh
