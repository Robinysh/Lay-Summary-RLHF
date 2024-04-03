import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


def decoder_collate(batch, tokenizer):
    queries, articles, article_ids = zip(*batch)
    inputs = tokenizer.batch_encode_plus(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    encoded_sents = [
        tokenizer.encode(
            sent, padding=True, truncation=False, return_tensors="pt"
        ).squeeze(0)
        for sent in queries
    ]

    lengths = torch.Tensor([len(data) for data in encoded_sents]).long()
    # ic(encoded_sents[0])
    # ic(inputs['input_ids'])
    # pad_id = tokenizer.encode("</s>")[-1]
    # inputs['input_ids'][inputs['input_ids'] == pad_id] = -100
    return inputs, lengths, encoded_sents, queries, articles, article_ids


class AbstractDataset(Dataset):
    def __init__(self, article_fpaths):
        self.articles = []
        for path in article_fpaths:
            self.articles += self.load_articles(path)

    def load_articles(self, fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for item in data:
            sections = item["article"].split("\n")
            item["sections"] = dict(zip(item["headings"], sections))
        return data

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]

        query = f"""<s>[INST]
        Rephrase the following abstract from a medical paper to make it more accessible and understandable to non-expert audiences, commonly referred to as "lay summaries".
        These lay summaries aim to present the key information from the original articles in a way that is less technical and contains more background information,
        making it easier for a broader range of readers, including researchers, medical professionals, journalists, and the general public, to comprehend the content.

        You should keep the words simple and the sentences short.
        You should also avoid using jargon and technical terms.
        The summary shold be easily understood by a 12th grade student.
        You can do so by replacing complex words with simpler synonyms (i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex sentence into several simpler ones.

        This is very important to my career. You'd better be sure.

        Abstract: {article['sections']['Abstract']}
        Lay summary of abstract: [/INST]"""

        return query, article, article["id"]


class AbstractDataModule(L.LightningDataModule):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        articles_fpath,
        batch_size: int = 1,
        tokenizer=None,
        dataset=AbstractDataset,
        collate_fn=None,
    ):
        super().__init__()
        self.articles_fpath = Path(articles_fpath)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collate_fn = collate_fn

    def train_dataloader(self):
        dataset = self.dataset(
            [self.articles_fpath / x for x in ["eLife_train.jsonl", "PLOS_train.jsonl"]]
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.collate_fn(x, self.tokenizer),
            num_workers=8,
        )

    def val_dataloader(self):
        dataset = self.dataset(
            [self.articles_fpath / x for x in ["eLife_val.jsonl", "PLOS_val.jsonl"]]
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.collate_fn(x, self.tokenizer),
            num_workers=8,
        )
