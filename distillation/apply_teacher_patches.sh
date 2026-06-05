#!/bin/bash
# Apply teacher patches from distillation/teacher_patches to teacher-model

set -e
PROJECT_DIR="/home/export/sdurgut/scratch/L1DeepMETv2-distillation"

if [ ! -d "${PROJECT_DIR}/teacher-model" ]; then
    echo "teacher-model directory not found! Please clone it first."
    exit 1
fi

cp "${PROJECT_DIR}/distillation/teacher_patches/net.py" "${PROJECT_DIR}/teacher-model/model/net.py"
cp "${PROJECT_DIR}/distillation/teacher_patches/graph_met_network.py" "${PROJECT_DIR}/teacher-model/model/graph_met_network.py"

echo "Teacher model successfully patched to return intermediate embeddings!"
