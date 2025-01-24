from huggingface_hub import snapshot_download
from paths import DATASET_DIR


def main():

    snapshot_download(
        repo_id="shuchangtao/CONQORD_dataset", 
        repo_type="dataset", 
        local_dir=DATASET_DIR,
    )


if __name__ == '__main__':
    main()
