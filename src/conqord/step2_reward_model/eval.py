import argparse
import os
import torch

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.data.data_utils import get_raw_dataset, DataCollatorReward, PromptDataset, create_prompt_dataset
from datasets import concatenate_datasets

import deepspeed

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Proper evaluation of the finetued reward model on a given dataset")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the dataset for evaluation')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store results.")
    args = parser.parse_args()
    return args

def evaluation_reward(model, eval_dataloader, device):
        model.eval()
        chosen_scores = []
        rejected_scores = []
        mean_loss = 0
        print(len(eval_dataloader))
        for step, batch in tqdm(enumerate(eval_dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            mean_loss += outputs["loss"].item()
            chosen_scores.extend(outputs["chosen_mean_scores"].tolist())
            rejected_scores.extend(outputs["rejected_mean_scores"].tolist())
            
        return chosen_scores, rejected_scores, mean_loss


def main():
    args = parse_args()
    
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    args.global_rank = torch.distributed.get_rank()
    set_random_seed(args.seed)
    torch.distributed.barrier()
    
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   None,
                                   args.num_padding_at_beginning,
                                   disable_dropout=True)
    rm_model.to(device)

    raw_dataset = get_raw_dataset(args.data_path, args.data_output_path, args.seed, args.local_rank)
    raw_eval_dataset = raw_dataset.get_eval_data()

    _, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, 2, args.seed, tokenizer,
        args.max_seq_len)
    print(_.shape, eval_dataset.shape)
        
    data_collator = DataCollatorReward()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    
    chosen_scores, rejected_scores, mean_loss = evaluation_reward(rm_model, eval_dataloader, device)
    print('Loss: ', mean_loss)

    raw_eval_dataset = raw_eval_dataset.add_column("chosen_scores", chosen_scores)
    raw_eval_dataset = raw_eval_dataset.add_column("rejected_scores", rejected_scores)   
    raw_eval_dataset.to_csv(f"{args.output_dir}/resulting_eval_dataset.csv")
    raw_eval_dataset.save_to_disk(f"{args.output_dir}/resulting_eval_dataset")

if __name__ == "__main__":
    main()
