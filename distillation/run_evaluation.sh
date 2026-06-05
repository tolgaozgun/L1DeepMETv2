#!/bin/bash
#SBATCH --job-name=distillation_eval
#SBATCH --output=distillation/logs/eval_%j.out
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --qos=medium
#SBATCH --gres=mps:50

# ── Environment setup ──────────────────────────────────────────────────────────
export TMPDIR="/home/export/sdurgut/scratch/tmp"
mkdir -p "$TMPDIR"
export CONDA_PKGS_DIRS="/home/export/sdurgut/scratch/mamba_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

set -euo pipefail

PROJECT_DIR="/home/export/sdurgut/scratch/L1DeepMETv2-distillation"
OUTPUT_DIR="${PROJECT_DIR}/distillation/outputs"
LOG_DIR="${PROJECT_DIR}/distillation/logs"

cd "${PROJECT_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "/home/export/sdurgut/scratch/mamba_envs/l1deepmet_distill_env"

# ── Argument: optionally pass a specific job output directory ──────────────────
# Usage:
#   sbatch distillation/run_evaluation.sh                   # evaluates ALL runs under distillation/outputs/
#   sbatch distillation/run_evaluation.sh job_17576         # evaluates a specific job folder only
#
FILTER_JOB="${1:-}"   # empty = evaluate all

echo "======================================================"
echo "Distillation Evaluation Job: ${SLURM_JOB_ID}"
echo "Scanning for completed checkpoints in: ${OUTPUT_DIR}"
echo "======================================================"

# Map output directory name fragment → student option argument
# Directory names look like: option_depth4_spaced, option_depth4_endpoints, option_depth2_matched
declare -A OPTION_MAP=(
    ["option_depth4_spaced"]="depth4_spaced"
    ["option_depth4_endpoints"]="depth4_endpoints"
    ["option_depth2_matched"]="depth2_matched"
)

eval_log_dir="${LOG_DIR}/eval_job_${SLURM_JOB_ID}"
mkdir -p "${eval_log_dir}"

n_evaluated=0
n_skipped=0

# Walk every leaf directory that contains a best.pth.tar
# Expected structure: distillation/outputs/job_XXXXX/ckpts_DATASET/option_OPTION/best.pth.tar
while IFS= read -r -d '' ckpt_file; do
    run_dir="$(dirname "$ckpt_file")"

    # If user filtered to a specific job, skip other jobs
    if [[ -n "$FILTER_JOB" ]] && [[ "$run_dir" != *"$FILTER_JOB"* ]]; then
        continue
    fi

    # Extract the option subdirectory name (last path component starting with "option_")
    option_dir="$(basename "$run_dir")"

    if [[ -z "${OPTION_MAP[$option_dir]+_}" ]]; then
        echo "[SKIP] Cannot determine --option for: $run_dir (unrecognised dir name: $option_dir)"
        ((n_skipped++)) || true
        continue
    fi

    option="${OPTION_MAP[$option_dir]}"

    # Skip if best.resolutions already exists (re-run with --force to override)
    if [[ -f "${run_dir}/best.resolutions" ]] && [[ "${FORCE_REEVAL:-0}" != "1" ]]; then
        echo "[SKIP] Already evaluated (best.resolutions exists): $run_dir"
        ((n_skipped++)) || true
        continue
    fi

    # Build a clean log file name from the run path
    log_name="$(echo "$run_dir" | sed 's|.*/distillation/outputs/||' | tr '/' '_')"
    eval_log="${eval_log_dir}/${log_name}_eval.log"

    echo "------------------------------------------------------"
    echo "Evaluating: $run_dir"
    echo "  option : $option"
    echo "  log    : $eval_log"
    echo "------------------------------------------------------"

    python evaluate_student.py \
        --ckpts    "$run_dir" \
        --option   "$option"  \
        --data     "data_ttbar" \
        --batch_size 32 \
        > "$eval_log" 2>&1
    exit_code=$?
    # Check whether the key output file was actually produced, not just the exit code.
    # NFS cleanup OSErrors can cause a non-zero exit even on success.
    if [[ -f "${run_dir}/best.resolutions" ]]; then
        echo "[DONE]  $run_dir"
    else
        echo "[ERROR] $run_dir (exit_code=${exit_code}) — see $eval_log"
    fi

    ((n_evaluated++)) || true

done < <(find "${OUTPUT_DIR}" -name "best.pth.tar" -print0)

echo "======================================================"
echo "Evaluation complete."
echo "  Runs evaluated : ${n_evaluated}"
echo "  Runs skipped   : ${n_skipped}"
echo ""
echo "To force re-evaluation of already-evaluated runs:"
echo "  FORCE_REEVAL=1 sbatch distillation/run_evaluation.sh"
echo ""
echo "Each run directory now contains best.resolutions."
echo "Point compare_response_resolution.ipynb at:"
echo "  ${OUTPUT_DIR}"
echo "======================================================"
