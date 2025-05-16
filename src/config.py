from enum import Enum


class SupportedModels(str, Enum):
    llama3_1b = "luezzka/Llama-3.2-1B-Instruct"
    gemma_2b = "google/gemma-2b-it"
    qwen = "Qwen/Qwen3-1.7B"
