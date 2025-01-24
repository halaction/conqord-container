from huggingface_hub import snapshot_download
from paths import MODEL_DIR


def main():

    snapshot_download(
        repo_id="luezzka/LLama-3.2-1B", 
        repo_type="model", 
        local_dir=MODEL_DIR / 'llama3_1b',
    )


if __name__ == '__main__':
    main()
