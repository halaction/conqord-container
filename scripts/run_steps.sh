#!/bin/sh

export HOME="/workspace"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export BASE_DIR="$HOME/conqord-container"
export SCRIPTS_DIR="$BASE_DIR/scripts"
export CONQORD_DIR="$BASE_DIR/src/conqord"
export HF_HUB_ENABLE_HF_TRANSFER=1

set -e

sh "$CONQORD_DIR/step1_supervised_finetuning_LM/run_step1.sh"
sh "$CONQORD_DIR/step2_reward_model/run_step2.sh"
sh "$CONQORD_DIR/step3_RL_finetune_LLM/run_step3.sh"
sh "$CONQORD_DIR/step4_evaluation/run_step4.sh"
