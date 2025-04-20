python3 baseline.py \
    --dataset VityaVitalich/adaptive_rag_natural_questions \
    --model google/gemma-2b-it \
    --batch_size 32 \
    --device cuda:0
    --output_folder results/models--google--gemma-2b-it

python3 baseline.py \
    --dataset VityaVitalich/adaptive_rag_natural_questions \
    --model halaction/conqord-checkpoint-step1 \
    --batch_size 32 \
    --device cuda:0
    --output_folder results/models--halaction--conqord-checkpoint-step1