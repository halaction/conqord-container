# AlignScore

AlignScore is a metric, based on RoBERTa, which evaluates the consistency of information between two texts, which are called a context and a claim.

## How to launch

```
python ./alignscore/alignscore.py \
    --dataset path/to/dataset \
    --context context \
    --claim claim \
    --model roberta-base \
    --batch_size 32 \
    --device cuda:0 \
    --ckpt_path ./alignscore/ckpt/ \
    --evaluation_mode nli_sp \
    --output ./alignscore/results.csv
```

`dataset` is a .csv file, which must contain two columns with names specified in `context` and `claim`.

`model` is either `roberta-base` or `roberta-large`.

`ckpt_path` is a path to a folder, containing one or both checkpoints:

1. AlignScore-base: https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt

2. AlignScore-large: https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt

They will be downloaded automatically if not present, but it might take a while.

`evaluation_mode` is one of `nli_sp`, `nli`, `bin_sp`, `bin`. `nli` and `bin` refer to different classification heads. `sp` indicates whether to use the original splitting method (claim is split into sentences, context into chunks to fit context window). The default (and the best in performance) is `nli_sp`.
