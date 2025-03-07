mkdir -p results
mkdir -p log

# Step 2.1: Downloading dataset from https://huggingface.co/datasets/hooope/CONQORD_datasets/conqord_step2_data, and save them to ../datasets/conqord_step2_data/

# Step 2.2: Run main.py in step2
export CUDA_VISIBLE_DEVICES=0
# nohup 
deepspeed --master_port 23001 eval.py \
   --data_path openai/webgpt_comparisons \
   --data_split 0,10,0 \
   --model_name_or_path checkpoint/step2/ \
   --data_output_path ../datasets/datatmp/ \
   --per_device_eval_batch_size 10 \
   --max_seq_len 512 \
   --num_padding_at_beginning 0 \
   --seed 1234 \
   --output_dir results/ \
#   &> log/step2_eval.log 2>&1 &

