# !bash ./get_lens.sh
# !pip install torchtext


import os, json
import argparse
import numpy as np
from lens.lens_score import LENS
import torch
import wandb


def calc_lens(preds, refs, docs):
    model_path = "./models/LENS/LENS/checkpoints/epoch=5-step=6102.ckpt"
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
    refs_dicts = read_file_lines(gold_path)
    preds = read_file_lines(pred_path)

    refs = [d["lay_summary"] for d in refs_dicts]
    docs = [d["article"] for d in refs_dicts]

    score_dict = {}

    score_dict[f"{dataset_name}_LENS"] = calc_lens(preds, refs, docs)
    wandb.log(score_dict)
    return score_dict


def write_scores(score_dict, output_filepath):
    with open(output_filepath, "w") as f:
        for key, value in score_dict.items():
            f.write(f"{key}: {value}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit_dir", type=str)
    parser.add_argument("--truth_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_run_id", type=str)
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        resume="allow"
    )

    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate + write eLife scores
    elife_scores = evaluate(
        os.path.join(submit_dir, "elife.txt"),
        os.path.join(truth_dir, "eLife_val.jsonl"),
        "eLife-val"
    )
    write_scores(elife_scores, os.path.join(output_dir, "elife_scores2.txt"))

    torch.cuda.empty_cache()

    # Calculate + write PLOS scores
    plos_scores = evaluate(
        os.path.join(submit_dir, "plos.txt"),
        os.path.join(truth_dir, "PLOS_val.jsonl"),
        "PLOS-val"
    )
    write_scores(plos_scores, os.path.join(output_dir, "plos_scores2.txt"))

    # Calculate + write overall scores
    avg_scores = {}
    metrics = ["LENS"]
    for metric in metrics:
        avg_scores[f"{metric}-val"] = np.mean([elife_scores[f"eLife-val_{metric}"], plos_scores[f"PLOS-val_{metric}"]])

    write_scores(avg_scores, os.path.join(output_dir, "scores2.txt"))
    wandb.finish()
