import argparse
import nltk
import os
import pandas as pd
import requests
from tqdm import tqdm
from typing import List

from .alignscore_package import AlignScore

ckpt_url = {
    'roberta-base': "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt",
    'roberta-large': "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"
}
ckpt_name = {
    'roberta-base': "AlignScore-base.ckpt",
    'roberta-large': "AlignScore-large.ckpt"
}

def download_checkpoint(url, save_path):
    print('Downloading checkpoint', flush=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192

    with open(save_path, "wb") as f, tqdm(
        desc=save_path,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print('Downloaded succesfully', flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Apply AlignScore metric to a dataset.")

    parser.add_argument("--dataset", type=str, 
        default="./alignscore/demonstrations.csv", 
        help="Path to dataset with columns context and claim")
    parser.add_argument("--context", type=str, 
        default="context", 
        help="Name for the context column in the dataset")
    parser.add_argument("--claim", type=str, 
        default="claim", 
        help="Name for the claim column in the dataset")
    parser.add_argument("--model", type=str, 
        default="roberta-base", 
        help="One of roberta-base or roberta-large")
    parser.add_argument("--batch_size", type=int, 
        default=32, 
        help="Batch size")
    parser.add_argument("--device", type=str, 
        default="cuda:0", 
        help="Device")
    parser.add_argument("--ckpt_path", type=str, 
        default="./alignscore/ckpt/", 
        help="Path to the folder with model checkpoint.")
    parser.add_argument("--evaluation_mode", type=str, 
        default="nli_sp", 
        help="One of: nli_sp, nli, bin_sp, bin")
    parser.add_argument("--output", type=str,
        default="./alignscore/results.csv",
        help="Path to output file")

    return parser.parse_args()

def alignscore(context : List[str],
               claim : List[str], 
               batch_size=32, 
               device="cuda:0", 
               evaluation_mode='nli_sp', 
               model="roberta-base", 
               ckpt_folder_path="./alignscore/ckpt/"
    ):
    url = ckpt_url[model]
    ckpt_path = os.path.join(ckpt_folder_path, ckpt_name[model])

    if not os.path.isfile(ckpt_path):
        os.makedirs(ckpt_folder_path, exist_ok=True)
        download_checkpoint(url, ckpt_path)
    
    nltk.download('punkt_tab')

    scorer = AlignScore(model=model,
        batch_size=batch_size, 
        device=device, 
        ckpt_path=ckpt_path,
        evaluation_mode=evaluation_mode)

    return scorer.score(contexts=context, claims=claim)

def main():
    args = parse_args()
    df = pd.read_csv(args.dataset)

    df['score'] = alignscore(
        list(df[args.context]),
        list(df[args.claim]),
        evaluation_mode=args.evaluation_mode,
        batch_size=args.batch_size, 
        device=args.device,
        model=args.model, 
        ckpt_folder_path=args.ckpt_path
    )

    df.to_csv(args.output)

if __name__ == "__main__":
    main()
