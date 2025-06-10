#!/bin/sh

export HOME="/workspace"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export BASE_DIR="$HOME/conqord-container"
export SCRIPTS_DIR="$BASE_DIR/scripts"
export EVALUATION_DIR="$BASE_DIR/src/evaluation"
export RESULTS_DIR="$EVALUATION_DIR/results"
export HF_HUB_ENABLE_HF_TRANSFER=1

set -e

ORGANIZATION_NAME=$(echo "$MODEL" | cut -d "/" -f 1)
MODEL_NAME=$(echo "$MODEL" | cut -d "/" -f 2)

DATASET="halaction/adaptive-rag-natural-questions"
MODEL_STEP3="halaction/$MODEL_NAME-conqord-step3-actor"

mkdir -p "$RESULTS_DIR"

python3 "$EVALUATION_DIR/baseline.py" \
    --dataset $DATASET \
    --model $MODEL_STEP3 \
    --batch_size 16 \
    --device cuda:0 \
    --output_folder "$RESULTS_DIR/$MODEL_NAME"
