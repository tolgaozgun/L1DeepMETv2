#!/bin/bash
#SBATCH --job-name=distillation_compare
#SBATCH --output=distillation/logs/compare_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --qos=medium

# ── Environment setup ──────────────────────────────────────────────────────────
export TMPDIR="/home/export/sdurgut/scratch/tmp"
mkdir -p "$TMPDIR"
export CONDA_PKGS_DIRS="/home/export/sdurgut/scratch/mamba_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

set -euo pipefail

PROJECT_DIR="/home/export/sdurgut/scratch/L1DeepMETv2-distillation"

cd "${PROJECT_DIR}"

source ~/scratch/ttHbb_SPANet/envs/notebook_env/bin/activate

echo "======================================================"
echo "Distillation Comparison Job: ${SLURM_JOB_ID}"
echo "======================================================"

python3 distillation/compare_distillation_results.py

echo "======================================================"
echo "Comparison complete."
echo "Plots saved to: ${PROJECT_DIR}/distillation/outputs/comparison_plots/"
echo "======================================================"
