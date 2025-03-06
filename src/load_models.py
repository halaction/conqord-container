import argparse

from huggingface_hub import snapshot_download

from paths import MODEL_DIR
from config import SupportedModels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='llama3_1b', type=str)
    return parser.parse_args()


def main(model_id):
    snapshot_download(
        repo_id=SupportedModels[model_id], 
        repo_type="model", 
        local_dir=MODEL_DIR / model_id,
    )


if __name__ == '__main__':
    args = parse_args()
    main(args.model_id)
