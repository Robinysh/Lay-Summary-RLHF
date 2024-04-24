# pylint: skip-file

import argparse
import json
import os

import numpy as np
import textstat
import torch
from alignscore import AlignScore
from bert_score import score
from lens.lens_score import LENS
from rouge_score import rouge_scorer
from summac.model_summac import SummaCConv

import wandb


def calc_rouge(preds, refs):
    # Get ROUGE F1 scores
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True
    )
    scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
    return (
        np.mean([s["rouge1"].fmeasure for s in scores]),
        np.mean([s["rouge2"].fmeasure for s in scores]),
        np.mean([s["rougeLsum"].fmeasure for s in scores]),
    )


def calc_bertscore(preds, refs):
    print("Calculating BertScore...")
    # Get BERTScore F1 scores
    P, R, F1 = score(preds, refs, lang="en", verbose=True, device="cuda:0")
    return np.mean(F1.tolist())


def calc_readability(preds):
    fkgl_scores = []
    cli_scores = []
    dcrs_scores = []
    for pred in preds:
        fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
        cli_scores.append(textstat.coleman_liau_index(pred))
        dcrs_scores.append(textstat.dale_chall_readability_score(pred))
    return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)


def calc_alignscore(preds, docs):
    print("Calculating AlignScore...")
    alignscorer = AlignScore(
        model="distilroberta-base",
        batch_size=8,
        device="cuda:0",
        ckpt_path="/data/colx531/eval_models/AlignScore.ckpt",
        evaluation_mode="nli_sp",
    )
    return np.mean(alignscorer.score(contexts=docs, claims=preds))


def calc_summac(preds, docs):
    print("Calculating SummaC...")
    model_conv = SummaCConv(
        models=["vitc-base"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device="cuda",
        start_file="default",
        agg="mean",
    )
    return np.mean(model_conv.score(docs, preds)["scores"])


def calc_lens(preds, refs, docs, model_path=None):
    print("Calculating LENS...")
    if model_path is None:
        model_path = "/data/colx531/eval_models/LENS/checkpoints/epoch=5-step=6102.ckpt"
    metric = LENS(model_path, rescale=True)
    abstracts = [d.split("\n")[0] for d in docs]
    refs = [[x] for x in refs]

    scores = metric.score(abstracts, preds, refs, batch_size=8, gpus=1)
    return np.mean(scores)


def read_file_lines(path):
    with open(path, "r") as f:
        lines = f.readlines()

    if path.endswith(".jsonl"):
        lines = [json.loads(line) for line in lines]

    return lines


def evaluate(pred_path, gold_path, dataset_name):
    # Load data from files
    refs_dicts = read_file_lines(gold_path)
    preds = read_file_lines(pred_path)

    refs_dicts = refs_dicts[: len(preds)]

    refs = [d["lay_summary"] for d in refs_dicts]
    docs = [d["article"] for d in refs_dicts]

    score_dict = {}

    # Relevance scores
    rouge1_score, rouge2_score, rougel_score = calc_rouge(preds, refs)
    score_dict[f"{dataset_name}_ROUGE1"] = rouge1_score
    score_dict[f"{dataset_name}_ROUGE2"] = rouge2_score
    score_dict[f"{dataset_name}_ROUGEL"] = rougel_score
    score_dict[f"{dataset_name}_BERTScore"] = calc_bertscore(preds, refs)

    # # Readability scores
    fkgl_score, cli_score, dcrs_score = calc_readability(preds)
    score_dict[f"{dataset_name}_FKGL"] = fkgl_score
    score_dict[f"{dataset_name}_DCRS"] = dcrs_score
    score_dict[f"{dataset_name}_CLI"] = cli_score

    # Factuality scores
    score_dict[f"{dataset_name}_AlignScore"] = calc_alignscore(preds, docs)
    score_dict[f"{dataset_name}_SummaC"] = calc_summac(preds, docs)
    score_dict[f"{dataset_name}_LENS"] = calc_lens(preds, refs, docs)
    wandb.log(score_dict)

    return score_dict


def write_scores(score_dict, output_filepath):
    # Write scores to file
    with open(output_filepath, "w") as f:
        for key, value in score_dict.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    # nltk.download("punkt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--submit_dir", type=str)
    parser.add_argument(
        "--truth_dir", type=str, default="/data/colx531/biolaysumm2024_data"
    )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--wandb_project", type=str, default="biolaysummary")
    parser.add_argument("--wandb_entity", type=str, default="colx-531-team-burrito")
    parser.add_argument("--wandb_name", type=str)
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        resume="allow",
    )

    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Calculate + write eLife scores
    elife_scores = evaluate(
        os.path.join(submit_dir, "elife.txt"),
        os.path.join(truth_dir, "eLife_val.jsonl"),
        "eLife-val",
    )
    write_scores(elife_scores, os.path.join(output_dir, "elife_scores1.txt"))

    torch.cuda.empty_cache()

    # Calculate + write PLOS scores
    plos_scores = evaluate(
        os.path.join(submit_dir, "plos.txt"),
        os.path.join(truth_dir, "PLOS_val.jsonl"),
        "PLOS-val",
    )
    write_scores(plos_scores, os.path.join(output_dir, "plos_scores1.txt"))

    # Calculate + write overall scores
    avg_scores = {}
    metrics = [
        "ROUGE1",
        "ROUGE2",
        "ROUGEL",
        "BERTScore",
        "FKGL",
        "DCRS",
        "CLI",
        "AlignScore",
        "SummaC",
        "LENS",
    ]
    for metric in metrics:
        avg_scores[f"{metric}-val"] = np.mean(
            [elife_scores[f"eLife-val_{metric}"], plos_scores[f"PLOS-val_{metric}"]]
        )
    write_scores(avg_scores, os.path.join(output_dir, "scores1.txt"))
    wandb.finish()
