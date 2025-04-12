#!/bin/sh

export HOME="/workspace"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export BASE_DIR="$HOME/conqord-container"
export SCRIPTS_DIR="$BASE_DIR/scripts"
export CONQORD_DIR="$BASE_DIR/src/conqord"
export MODEL_ID="${MODEL_ID:-google/gemma-2b-it}"

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Downloading $MODEL_ID..."
huggingface-cli download $MODEL_ID

echo "Running step 1..."
cd "$CONQORD_DIR/step1_supervised_finetuning_LM"
sh run_step1.sh

echo "Uploading step 1 checkpoint..."
REPO_ID="halaction/conqord-checkpoint-step1"
LOCAL_PATH="./checkpoint/step1"
huggingface-cli upload $REPO_ID $LOCAL_PATH . --repo-type model

# echo "Running step 2..."
# cd "$SOURCE_DIR/conqord/step2_reward_model"
# sh run_step2.sh

# echo "Running step 3..."
# cd "$SOURCE_DIR/conqord/step3_RL_finetune_LLM"
# sh run_step3.sh
