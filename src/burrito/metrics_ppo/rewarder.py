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


# def calc_bertscore(preds, refs):
#     print("Calculating BertScore...")
#     # Get BERTScore F1 scores
#     P, R, F1 = bert_score.score(preds, refs, lang="en", verbose=True, device="cuda:0", model_type="microsoft/deberta-base-mnli", use_fast_tokenizer=True)
#     return F1.tolist()


def calc_readability(preds):
    fkgl_scores = []
    cli_scores = []
    dcrs_scores = []
    for pred in preds:
        fkgl_scores.append(flesch_kincaid_grade(pred))  #
        cli_scores.append(coleman_liau_index(pred))
        dcrs_scores.append(dale_chall_readability_score(pred))
    return fkgl_scores, cli_scores, dcrs_scores


def calc_alignscore(preds, docs):
    print("Calculating AlignScore...")
    alignscorer = AlignScore(
        model="distilroberta-base",
        batch_size=8,
        device="cuda:0",
        ckpt_path="/data/colx531/eval_models/AlignScore.ckpt",
        evaluation_mode="nli_sp",
    )
    return alignscorer.score(contexts=docs, claims=preds)


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
    return model_conv.score(docs, preds)["scores"]


def calc_lens(preds, refs, docs, model_path=None):
    print("Calculating LENS...")
    if model_path is None:
        model_path = "/data/colx531/eval_models/LENS/checkpoints/epoch=5-step=6102.ckpt"
    metric = LENS(model_path, rescale=True)
    abstracts = [d.split("\n")[0] for d in docs]
    refs = [[x] for x in refs]

    scores = metric.score(abstracts, preds, refs, batch_size=8, gpus=1)
    return scores


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

        # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        reward = [
            (
                score_dict["ROUGE1"][i]
                + score_dict["ROUGE2"][i]
                + score_dict["ROUGEL"][i]
                + score_dict["BERTScore"][i]
                - score_dict["FKGL"][i]
                - score_dict["DCRS"][i]
                - score_dict["CLI"][i]
                + score_dict["LENS"][i]
                + score_dict["AlignScore"][i]
                + score_dict["SummaC"][i]
            )
            / len(score_dict)
            for i in range(len(preds))
        ]
        return reward, score_dict
