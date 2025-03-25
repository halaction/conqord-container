#!/bin/sh

# Step2 Obtaining reward model
# The current directory is ./step2_reward_model/ (=CONQORD/step2_reward_model/)

# Step 2.0: Create log, checkpoint, tensorboard folders
mkdir -p log
mkdir -p checkpoint
mkdir -p tensorboard

# Step 2.1: Downloading dataset from https://huggingface.co/datasets/hooope/CONQORD_datasets/conqord_step2_data, and save them to ../datasets/conqord_step2_data/
# Step 2.2: Run main.py in step2
export CUDA_VISIBLE_DEVICES=0,1,2 
deepspeed --master_port 23001 main.py \
   --data_path openai/webgpt_comparisons \
   --data_split 0,1,10 \
   --model_name_or_path ../model_pth/gemma_2b/ \
   --data_output_path ../datasets/datatmp/ \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 128 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 64 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --offload \
   --deepspeed \
   --lora_dim 64 \
   --lora_module_name "layers." \
   --output_dir checkpoint/step2 \
   --enable_tensorboard \
   --tensorboard_path tensorboard/step2 \
   --debug \
   > log/step2.log 2>&1








