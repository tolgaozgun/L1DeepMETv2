#!/bin/bash
#SBATCH --job-name=student_baseline
#SBATCH --output=distillation/logs/baseline_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --qos=medium
#SBATCH --gres=mps:50

nvidia-smi

# Redirect temporary files to scratch to prevent out-of-space errors on compute nodes
export TMPDIR="/home/export/sdurgut/scratch/tmp"
mkdir -p "$TMPDIR"
export CONDA_PKGS_DIRS="/home/export/sdurgut/scratch/mamba_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

set -euo pipefail

PROJECT_DIR="/home/export/sdurgut/scratch/L1DeepMETv2-distillation"
LOG_DIR="${PROJECT_DIR}/distillation/logs"
OUTPUT_DIR="${PROJECT_DIR}/distillation/outputs/baselines"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "/home/export/sdurgut/scratch/mamba_envs/l1deepmet_distill_env"

STUDENT_OPTIONS=("depth4_spaced" "depth4_endpoints" "depth2_matched")
DATA_DIR="data_ttbar"
BATCH_SIZE=32
LR=0.1
WEIGHT_DECAY=0.001

MAX_JOBS=3
echo "Starting concurrent baseline training (Max $MAX_JOBS at a time)..."

for option in "${STUDENT_OPTIONS[@]}"; do
    echo "=========================================================="
    echo "Queuing baseline training for Option: $option"
    echo "=========================================================="
    
    run_output_dir="${OUTPUT_DIR}/job_${SLURM_JOB_ID}/option_${option}"
    mkdir -p "${run_output_dir}"

    run_log_dir="${LOG_DIR}/job_${SLURM_JOB_ID}"
    mkdir -p "${run_log_dir}"
    run_log_file="${run_log_dir}/baseline_option_${option}_job_${SLURM_JOB_ID}.log"

    python train_student_baseline.py \
        --option "$option" \
        --data "$DATA_DIR" \
        --ckpts "$run_output_dir" \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY > "$run_log_file" 2>&1 &
        
    if [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; then
        wait -n
    fi
done

echo "Waiting for all remaining jobs to finish..."
wait

echo "All baseline training completed successfully."
