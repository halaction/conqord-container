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

# Step 1
STEP_PATH="$CONQORD_DIR/step1_supervised_finetuning_LM"
REPO_ID="halaction/$MODEL_NAME-conqord-step1"
CHECKPOINT_PATH="$STEP_PATH/checkpoint/step1"

cd $STEP_PATH

echo "Running step 1..."
MODEL_PATH="$MODEL_PATH" sh run_step1.sh

rm -rf "$CHECKPOINT_PATH/pytorch_model.bin"
mv "$CHECKPOINT_PATH/final/pytorch_model.bin" "$CHECKPOINT_PATH/pytorch_model.bin"

echo "Uploading step 1 checkpoint..."
huggingface-cli upload $REPO_ID $CHECKPOINT_PATH . --repo-type model

# Step 2
STEP_PATH="$CONQORD_DIR/step2_reward_model"
cd $STEP_PATH

# echo "Running step 2..."
# sh run_step2.sh

# Step 3
STEP_PATH="$CONQORD_DIR/step3_RL_finetune_LLM"
cd $STEP_PATH

# echo "Running step 3..."
# sh run_step3.sh
