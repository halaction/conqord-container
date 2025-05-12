#!/bin/sh

export HOME="/workspace"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export BASE_DIR="$HOME/conqord-container"
export SCRIPTS_DIR="$BASE_DIR/scripts"
export CONQORD_DIR="$BASE_DIR/src/conqord"
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL="${MODEL:-google/gemma-2b-it}"
ORGANIZATION_NAME=$(echo "$MODEL" | cut -d "/" -f 1)
MODEL_NAME=$(echo "$MODEL" | cut -d "/" -f 2)

echo "Downloading $MODEL..."
MODEL_PATH=$(huggingface-cli download $MODEL)

echo "Running step 1..."
cd "$CONQORD_DIR/step1_supervised_finetuning_LM"
MODEL_PATH="$MODEL_PATH" sh run_step1.sh

echo "Uploading step 1 checkpoint..."
REPO_ID="halaction/$MODEL_NAME-conqord-step1"
LOCAL_PATH="./checkpoint/step1"
huggingface-cli upload $REPO_ID $LOCAL_PATH . --repo-type model

# echo "Running step 2..."
# cd "$SOURCE_DIR/conqord/step2_reward_model"
# sh run_step2.sh

# echo "Running step 3..."
# cd "$SOURCE_DIR/conqord/step3_RL_finetune_LLM"
# sh run_step3.sh
