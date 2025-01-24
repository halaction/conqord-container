from transformers import AutoTokenizer
from paths import MODEL_DIR


def main():

    # 
    tokenizer = AutoTokenizer.from_pretrained("luezzka/LLama-3.2-1B")
    tokenizer.save_pretrained('./')


if __name__ == '__main__':
    main()
