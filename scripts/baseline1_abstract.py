import json

import jsonlines
from tqdm import tqdm


def baseline_model(doc):
    return doc["sections"]["Abstract"]


if __name__ == "__main__":
    DATA_FOLDER = "/data/colx531/biolaysumm2024_data/"
    file_names = ["PLOS_test.jsonl", "eLife_test.jsonl"]

    for file_name in file_names:
        with open(DATA_FOLDER + file_name, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for item in data:
            sections = item["article"].split("\n")
            item["sections"] = dict(zip(item["headings"], sections))

        predictions = []
        for item in tqdm(data, leave=True):
            rephrased_abstract = baseline_model(item)
            predictions.append({"id": item["id"], "prediction": rephrased_abstract})

        with jsonlines.open(
            f'prediction_baseline1_Abstract_{file_name.split(".", maxsplit=1)[0]}.jsonl',
            mode="w",
        ) as writer:
            writer.write_all(predictions)
