import argparse
import os
import pandas as pd
import torch
from datasets import load_dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignscore import alignscore
from polygraph import LMPolygraph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply AlignScore metric to a dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="VityaVitalich/adaptive_rag_natural_questions",
        help="Path to dataset with columns context and claim",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="luezzka/Llama-3.2-1B-Instruct",
        help="One of roberta-base or roberta-large",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results/first",
        help="Path to output file",
    )
    parser.add_argument(
        "--results_output_name",
        type=str,
        default="results.csv",
        help="Path to output file with uncertainty and alignscore results",
    )
    parser.add_argument(
        "--metrics_output_name",
        type=str,
        default="metrics.csv",
        help="Path to output file with metrics",
    )
    parser.add_argument(
        "--alignscore_model",
        type=str,
        default="roberta-base",
        help="One of roberta-base or roberta-large",
    )
    parser.add_argument(
        "--alignscore_batch_size",
        type=int,
        default=16,
        help="Batch size for alignscore",
    )
    parser.add_argument(
        "--alignscore_ckpt_path",
        type=str,
        default="./alignscore/ckpt/",
        help="Path to the folder with model checkpoint.",
    )
    parser.add_argument(
        "--alignscore_evaluation_mode",
        type=str,
        default="nli_sp",
        help="One of: nli_sp, nli, bin_sp, bin",
    )

    return parser.parse_args()


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    polygraph = LMPolygraph(model, tokenizer, args.model)

    dataset = load_dataset(args.dataset)

    questions = dataset["train"]["question_text"]
    accepted_answers = dataset["train"]["reference"]
    for i in range(len(accepted_answers)):
        accepted_answers[i] = (
            ", ".join(accepted_answers[i])
            if isinstance(accepted_answers[i], list)
            else accepted_answers[i]
        )

    results = polygraph(questions, batch_size=args.batch_size, verbose=True)

    del polygraph, model, tokenizer

    alignscore_scores = alignscore(
        results["generated_text"],
        accepted_answers,
        evaluation_mode=args.alignscore_evaluation_mode,
        batch_size=args.alignscore_batch_size,
        device=args.device,
        model=args.alignscore_model,
        ckpt_folder_path=args.alignscore_ckpt_path,
    )

    results["question_text"] = questions
    results["accepted_answer"] = accepted_answers
    results["alignscore"] = alignscore_scores

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "question_text",
            "accepted_answer",
            "generated_text",
            "alignscore",
            "Perplexity",
            "MaximumSequenceProbability",
            "LexicalSimilarity_rougeL",
            "NumSemSets",
            "VerbalUncertainty",
            "VerbalUncertainty_corrected",
        ]
    ]
    results_df.to_csv(os.path.join(args.output_folder, args.results_output_name))

    metrics = pd.DataFrame(columns=["eu_estimate", "metric", "value"])
    for ue_estimate in [
        "Perplexity",
        "MaximumSequenceProbability",
        "LexicalSimilarity_rougeL",
        "NumSemSets",
        "VerbalUncertainty_corrected",
    ]:
        for metric in ["prr", "ece", "spearman_corr", "kendall_corr"]:
            metric_value = getattr(LMPolygraph, metric)(
                results[ue_estimate], alignscore_scores
            )
            _ = pd.DataFrame(
                [[ue_estimate, metric, metric_value]],
                columns=["eu_estimate", "metric", "value"],
            )
            metrics = pd.concat([metrics, _], ignore_index=True)

    metrics.to_csv(os.path.join(args.output_folder, args.metrics_output_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
