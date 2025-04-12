#!/bin/sh

export HOME="/workspace"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export BASE_DIR="$HOME/conqord-container"
export SCRIPTS_DIR="$BASE_DIR/scripts"
export CONQORD_DIR="$BASE_DIR/src/conqord"
export MODEL="${MODEL:-google/gemma-2b-it}"

sh "$SCRIPTS_DIR/download_model.sh"

cd "$CONQORD_DIR/step1_supervised_finetuning_LM"
sh run_step1.sh

# cd "$SOURCE_DIR/conqord/step2_reward_model"
# sh run_step2.sh

# cd "$SOURCE_DIR/conqord/step3_RL_finetune_LLM"
# sh run_step3.sh
