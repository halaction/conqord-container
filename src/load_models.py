from huggingface_hub import snapshot_download

from paths import MODEL_DIR
from config import SupportedModels


def main():

    model_id = 'llama3_1b'

    snapshot_download(
        repo_id=SupportedModels[model_id], 
        repo_type="model", 
        local_dir=MODEL_DIR / model_id,
    )


if __name__ == '__main__':
    main()
