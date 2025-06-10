#!/bin/sh

# Step 1: Supervised fine-tuning LLM to output confidence
set -e

export HOME="/workspace"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export BASE_DIR="$HOME/conqord-container"
export SCRIPTS_DIR="$BASE_DIR/scripts"
export CONQORD_DIR="$BASE_DIR/src/conqord"
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0,1,2

if [[ -z "$MODEL" ]]; then
   echo "Required variable MODEL is not set."
   exit 1

if [[ -z "$HF_TOKEN" ]]; then
   echo "Required variable HF_TOKEN is not set."
   exit 1

ORGANIZATION_NAME=$(echo "$MODEL" | cut -d "/" -f 1)
MODEL_NAME=$(echo "$MODEL" | cut -d "/" -f 2)

echo "Downloading $MODEL..."
MODEL_PATH=$(huggingface-cli download $MODEL)

STEP_PATH="$CONQORD_DIR/step1_supervised_finetuning_LM"
cd $STEP_PATH

mkdir -p log
mkdir -p checkpoint
mkdir -p tensorboard

echo "Running step 1 with $MODEL..."
deepspeed --master_port 13001 main.py \
   --data_path "halaction/adaptive-rag-natural-questions" \
   --data_split "10,0,0" \
   --model_name_or_path "$MODEL_PATH/" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --data_output_path "../datasets/datatmp/" \
   --max_seq_len 256 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 64 \
   --lr_scheduler_type "cosine" \
   --num_warmup_steps 5 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir "checkpoint/step1" \
   --print_loss \
   --enable_tensorboard \
   --tensorboard_path "tensorboard/step1" \
   > log/step1.log 2>&1

REPO_ID="halaction/$MODEL_NAME-conqord-step1"
CHECKPOINT_PATH="$STEP_PATH/checkpoint/step1"

# rm -rf "$CHECKPOINT_PATH/pytorch_model.bin"
# mv "$CHECKPOINT_PATH/final/pytorch_model.bin" "$CHECKPOINT_PATH/pytorch_model.bin"

# echo "Uploading step 1 checkpoint to $REPO_ID..."
# huggingface-cli upload $REPO_ID $CHECKPOINT_PATH . --repo-type model

echo "Finished step 1."
