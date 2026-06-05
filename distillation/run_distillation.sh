#!/bin/bash
#SBATCH --job-name=distillation_training
#SBATCH --output=distillation/logs/distillation_%j.out
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

# Redirect micromamba package cache to scratch to prevent lock and space errors
export CONDA_PKGS_DIRS="/home/export/sdurgut/scratch/mamba_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

set -euo pipefail

# Define project directories
PROJECT_DIR="/home/export/sdurgut/scratch/L1DeepMETv2-distillation"
LOG_DIR="${PROJECT_DIR}/distillation/logs"
OUTPUT_DIR="${PROJECT_DIR}/distillation/outputs"

# Create necessary directories
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"

ENV_PATH="/home/export/sdurgut/scratch/mamba_envs/l1deepmet_distill_env"
if [ ! -d "$ENV_PATH" ]; then
    echo "Environment does not exist. Creating it at $ENV_PATH..."
    micromamba create -y -p "$ENV_PATH" python=3.9
fi

micromamba activate "$ENV_PATH"

# Ensure necessary libraries are installed
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu121
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric
pip install coffea mplhep

# Verify Python environment
python - <<'PY'
import sys
import torch
import torch_geometric
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("TorchGeometric:", torch_geometric.__version__)
PY

# Verify GPU
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

# Define the datasets/checkpoints for the teacher
TEACHER_CKPTS=(
    "teacher-model/ckpts_dytt/best.pth.tar"
    "teacher-model/ckpts_znunu/best.pth.tar"
)

# Define the student model feature hint options
STUDENT_OPTIONS=("depth4_spaced" "depth4_endpoints" "depth2_matched")

# Training Hyperparameters matching run_train.py
BATCH_SIZE=32
LR=0.1
WEIGHT_DECAY=0.001


MAX_JOBS=1
echo "Starting concurrent knowledge distillation training (Max $MAX_JOBS at a time)..."

# Loop over each teacher checkpoint
for ckpt in "${TEACHER_CKPTS[@]}"; do
    dataset_name=$(basename $(dirname "$ckpt")) # e.g., ckpts_dytt
    # The student always trains on TTbar data regardless of which teacher checkpoint is used.
    # (teacher checkpoint names reflect the teacher's training physics process, not the student data)
    data_dir="data_ttbar"

    # Loop over each student option
    for option in "${STUDENT_OPTIONS[@]}"; do
        echo "=========================================================="
        echo "Queuing training for Dataset: $data_dir | Option: $option"
        echo "=========================================================="
        
        # Sub-directory for this specific combination with job ID to prevent overwriting
        run_output_dir="${OUTPUT_DIR}/job_${SLURM_JOB_ID}/${dataset_name}/option_${option}"
        mkdir -p "${run_output_dir}"

        # Define individual log file for this specific run
        run_log_dir="${LOG_DIR}/job_${SLURM_JOB_ID}"
        mkdir -p "${run_log_dir}"
        run_log_file="${run_log_dir}/${dataset_name}_option_${option}_job_${SLURM_JOB_ID}.log"

        # Run the training script from the project root in the background
        python train_distillation.py \
            --teacher_ckpt "$ckpt" \
            --option "$option" \
            --data "$data_dir" \
            --ckpts "$run_output_dir" \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --weight_decay $WEIGHT_DECAY > "$run_log_file" 2>&1 &
            
        # Wait if we have reached the maximum number of concurrent jobs
        if [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; then
            wait -n
        fi
    done
done

# Wait for any remaining background jobs to finish
echo "Waiting for all remaining jobs to finish..."
wait

echo "All training combinations completed successfully."
