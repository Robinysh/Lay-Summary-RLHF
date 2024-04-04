import bert_score
from alignscore import AlignScore
from lens.lens_score import LENS
from rouge_score import rouge_scorer
from summac.model_summac import SummaCConv

# pylint: disable-next=no-name-in-module
from textstat import (
    coleman_liau_index,
    dale_chall_readability_score,
    flesch_kincaid_grade,
)

from burrito.metrics_ppo.utils import time_it


def calc_rouge(preds, refs):
    # Get ROUGE F1 scores
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True
    )
    scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
    return (
        [s["rouge1"].fmeasure for s in scores],
        [s["rouge2"].fmeasure for s in scores],
        [s["rougeLsum"].fmeasure for s in scores],
    )


def calc_readability(preds):
    fkgl_scores = []
    cli_scores = []
    dcrs_scores = []
    for pred in preds:
        fkgl_scores.append(flesch_kincaid_grade(pred))  #
        cli_scores.append(coleman_liau_index(pred))
        dcrs_scores.append(dale_chall_readability_score(pred))
    return fkgl_scores, cli_scores, dcrs_scores


class Rewarder:
    def __init__(self):
        self.bert_scorer = bert_score.BERTScorer(
            model_type="microsoft/deberta-base-mnli",
            num_layers=9,
            lang="en",
            device="cuda:0",
            use_fast_tokenizer=True,
        )
        self.align_scorer = AlignScore(
            model="distilroberta-base",
            batch_size=8,
            device="cuda:0",
            ckpt_path="/data/colx531/eval_models/AlignScore.ckpt",
            evaluation_mode="nli_sp",
            verbose=False,
        )

        self.summac_scorer = SummaCConv(
            models=["vitc-base"],
            bins="percentile",
            granularity="sentence",
            nli_labels="e",
            device="cuda",
            start_file="default",
            agg="mean",
        )
        self.lens_scorer = LENS(
            "/data/colx531/eval_models/LENS/checkpoints/epoch=5-step=6102.ckpt",
            rescale=True,
        )

        # min max metric values from current and last years leaderbboard.
        self.metric_stats = {
            "BERTScore": [0.8255, 0.8707],
            "ROUGE1": [0.3653, 0.4946],
            "ROUGE2": [0.0882, 0.1676],
            "ROUGEL": [0.3362, 0.4615],
            "FKGL": [10.7, 15.75],
            "DCRS": [8.45, 11.77],
            "CLI": [13.25, 17.18],
            "LENS": [32.77, 74.67],
            "AlignScore": [0.7152, 0.9865],
            "SummaC": [0.5655, 0.9536],
        }

    def calc_alignscore(self, preds, docs):
        return self.align_scorer.score(contexts=docs, claims=preds)

    def calc_bertscore(self, preds, refs):
        _, _, f1 = self.bert_scorer.score(preds, refs)
        return f1.tolist()

    def calc_summac(self, preds, docs):
        return self.summac_scorer.score(docs, preds)["scores"]

    def calc_lens(self, preds, refs, docs):
        abstracts = [d.split("\n")[0] for d in docs]
        refs = [[x] for x in refs]
        scores = self.lens_scorer.score(
            abstracts, preds, refs, batch_size=8, gpus=1, verbose=False
        )
        return scores

    # pylint: disable=too-many-locals
    def __call__(self, preds, articles):
        score_dict = {}

        refs = [d["lay_summary"] for d in articles]
        docs = [d["article"] for d in articles]

        rouge1_score, rouge2_score, rougel_score = calc_rouge(preds, refs)
        score_dict["ROUGE1"] = rouge1_score
        score_dict["ROUGE2"] = rouge2_score
        score_dict["ROUGEL"] = rougel_score
        with time_it("bert"):
            score_dict["BERTScore"] = self.calc_bertscore(preds, refs)

        # # Readability scores
        fkgl_score, cli_score, dcrs_score = calc_readability(preds)
        score_dict["FKGL"] = fkgl_score
        score_dict["DCRS"] = dcrs_score
        score_dict["CLI"] = cli_score
        with time_it("lens"):
            score_dict["LENS"] = self.calc_lens(preds, refs, docs)

        # Factuality scores
        with time_it("alignscore"):
            score_dict["AlignScore"] = self.calc_alignscore(preds, docs)
        with time_it("summac"):
            score_dict["SummaC"] = self.calc_summac(preds, docs)

        norm_score_dict = {}
        for key, value in score_dict.items():
            norm_range = self.metric_stats[key][1] - self.metric_stats[key][0]
            norm_score_dict[key] = [
                (s - self.metric_stats[key][0]) / norm_range for s in value
            ]
        reward = [
            (
                norm_score_dict["ROUGE1"][i]
                + norm_score_dict["ROUGE2"][i]
                + norm_score_dict["ROUGEL"][i]
                + norm_score_dict["BERTScore"][i]
                + (1 - norm_score_dict["FKGL"][i])
                + (1 - norm_score_dict["DCRS"][i])
                + (1 - norm_score_dict["CLI"][i])
                + norm_score_dict["LENS"][i]
                + norm_score_dict["AlignScore"][i]
                + norm_score_dict["SummaC"][i]
            )
            / len(norm_score_dict)
            for i in range(len(preds))
        ]
        return reward, score_dict
