#!/bin/sh

cd /workspace/src
python3 load_datasets.py
python3 load_models.py

ls -a /workspace/src/conqord/datasets
ls -a /workspace/src/conqord/datasets

cd /workspace/src/conqord/step1_supervised_finetuning_LM
sh run_step1.sh

cd /workspace/src/conqord/step2_reward_model
sh run_step2.sh
