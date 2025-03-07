mkdir -p results

# Step 2.1: Downloading dataset from https://huggingface.co/datasets/hooope/CONQORD_datasets/conqord_step2_data, and save them to ../datasets/conqord_step2_data/

# Step 2.2: Run main.py in step2
export CUDA_VISIBLE_DEVICES=0,1,2
# nohup 
deepspeed --master_port 23001 eval.py \
   --data_path openai/webgpt_comparisons \
   --model_name_or_path ../model_pth/gemma_2b/ \
   --data_output_path ../datasets/datatmp/ \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --num_padding_at_beginning 0 \
   --seed 1234 \
   --deepspeed \
   --output_dir results/ \
   &> log/step2.log 2>&1 &

