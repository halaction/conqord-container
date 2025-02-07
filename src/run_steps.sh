#!/bin/sh

python3 load_models.py

cd conqord/step1_supervised_finetuning_LM
sh run_step1.sh
