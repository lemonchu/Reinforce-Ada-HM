#!/bin/bash
#SBATCH --job-name=mengchu_harmonic_eval_orz_b
#SBATCH --output=mengchu_b.out
#SBATCH --error=mengchu_b.err
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:RTX_PRO_6000:4

cd /data/group_data/cx_group/lemonpig/Reinforce-Ada
source .venv/bin/activate
bash scripts/run_reinforce_ada_fix.sh
sleep 300