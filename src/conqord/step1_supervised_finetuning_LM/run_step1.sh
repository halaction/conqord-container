#!/bin/sh

# Step 1: Supervised finetuning LLM to output confidence

MODEL_DIR="models--google--gemma-2b-it"
SNAPSHOT_HASH="96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"

mkdir -p log
mkdir -p checkpoint
mkdir -p tensorboard

export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --master_port 13001 main.py \
   --data_path openai/webgpt_comparisons \
   --data_split 10,0,0 \
   --model_name_or_path "$HF_HUB_CACHE/$MODEL_DIR/snapshots/$SNAPSHOT_HASH/" \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --data_output_path ../datasets/datatmp/ \
   --max_seq_len 256 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 64 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 5 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir checkpoint/step1 \
   --print_loss \
   --enable_tensorboard \
   --tensorboard_path tensorboard/step1 \
   > log/step1.log 2>&1
