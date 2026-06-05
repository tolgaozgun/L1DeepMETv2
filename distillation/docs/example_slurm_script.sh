#!/bin/bash
#SBATCH --job-name=classifier_btag_full_gpu
#SBATCH --output=logs/classifier/btag_full/classifier_btag_full_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --qos=medium
#SBATCH --gres=mps:50
nvidia-smi
set -euo pipefail
SPANET_DIR=/home/export/sdurgut/scratch/ttHbb_SPANet/SPANet

PROJECT_DIR="/home/export/sdurgut/scratch/ttHbb_SPANet"
OPTIONS_FILE="${PROJECT_DIR}/options_files/ttHbb_fully_hadronic/classifier/options_file_Run2_Run3_sig_QCD_classifier_btag_full.json"
LOG_DIR="${PROJECT_DIR}/spanet_output/classifier_btag_full"
VENV_ACTIVATE="${PROJECT_DIR}/SPANet/train_env/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${LOG_DIR}"
source "${VENV_ACTIVATE}"
echo "Activated venv: ${VENV_ACTIVATE}"
which python
python -V
export PYTHONPATH="${PROJECT_DIR}/SPANet:${PYTHONPATH:-}"
echo "PYTHONPATH=${PYTHONPATH}"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY

python -m spanet.train \
  -of "${OPTIONS_FILE}" \
  --log_dir "${LOG_DIR}" \
  --name "classifier_btag_full" \
  --time_limit 07:00:00:00 \
  --gpus 1 \
  --verbose \
  --fp16
