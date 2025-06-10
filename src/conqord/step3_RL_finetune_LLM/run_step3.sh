#!/bin/sh

# Step 3: RL fine-tune LM
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

MODEL_STEP1="halaction/$MODEL_NAME-conqord-step1"
MODEL_STEP2="halaction/$MODEL_NAME-conqord-step2"

echo "Downloading $MODEL_STEP1..."
MODEL_STEP1_PATH=$(huggingface-cli download $MODEL_STEP1)

echo "Downloading $MODEL_STEP2..."
MODEL_STEP2_PATH=$(huggingface-cli download $MODEL_STEP2)

STEP_PATH="$CONQORD_DIR/step3_RL_finetune_LLM"
cd $STEP_PATH

mkdir -p log
mkdir -p checkpoint
mkdir -p tensorboard

echo "Running step 3 with $MODEL..."
deepspeed --master_port 33001 main.py \
   --data_path "halaction/adaptive-rag-natural-questions" \
   --data_split "0,0,10" \
   --actor_model_name_or_path "$MODEL_STEP1_PATH" \
   --tokenizer_model_name_or_path "$MODEL_STEP1_PATH" \
   --critic_model_name_or_path "$MODEL_STEP2_PATH" \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 1e-6 \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type "constant" \
   --gradient_accumulation_steps 32 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir "checkpoint/step3" \
   --enable_tensorboard \
   --tensorboard_path "tensorboard/step3" \
   > log/step3.log 2>&1

REPO_ID="halaction/$MODEL_NAME-conqord-step3"
CHECKPOINT_PATH="$STEP_PATH/checkpoint/step3"

LAST_EPOCH=$(ls -d "$CHECKPOINT_PATH"/ep*/ | sort -V | tail -n 1)
LAST_EPOCH=${LAST_EPOCH%/}

LAST_STEP=$(ls -d "$CHECKPOINT_PATH/$LAST_EPOCH"/step*/ | sort -V | tail -n 1)
LAST_STEP=${LAST_STEP%/}

# rm -rf "$CHECKPOINT_PATH/actor/pytorch_model.bin"
# mv "$CHECKPOINT_PATH/$LAST_EPOCH/$LAST_STEP/actor/pytorch_model.bin" "$CHECKPOINT_PATH/actor/pytorch_model.bin"

# echo "Uploading step 3 checkpoint to $REPO_ID-actor..."
# huggingface-cli upload "$REPO_ID-actor" "$CHECKPOINT_PATH/actor" . --repo-type model

# rm -rf "$CHECKPOINT_PATH/critic/pytorch_model.bin"
# mv "$CHECKPOINT_PATH/$LAST_EPOCH/$LAST_STEP/critic/pytorch_model.bin" "$CHECKPOINT_PATH/critic/pytorch_model.bin"

# echo "Uploading step 3 checkpoint to $REPO_ID-critic..."
# huggingface-cli upload "$REPO_ID-critic" "$CHECKPOINT_PATH/critic" . --repo-type model

echo "Finished step 3."

