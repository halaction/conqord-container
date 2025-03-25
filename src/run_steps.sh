#!/bin/sh

cd /workspace/conqord-container/src
python3 load_models.py --model_id gemma_2b

cd /workspace/conqord-container/src/conqord/step1_supervised_finetuning_LM
sh run_step1.sh

cd /workspace/conqord-container/src/conqord/step2_reward_model
sh run_step2.sh

cd /workspace/conqord-container/src/conqord/step3_RL_finetune_LLM
sh run_step3.sh
