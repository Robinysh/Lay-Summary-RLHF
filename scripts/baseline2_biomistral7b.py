import json

import jsonlines
from openai import OpenAI
from tqdm import tqdm


def write_lines_to_file(file_path, lines):
    """
    Write a list of lines to a file, with each element of the list as a separate line in the file.

    :param file_path: str, the path to the file to be written to
    :param lines: list of str, the lines of text to write to the file
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


def rephrase_texts(text):
    q_prompt = """
    Rephrase the following abstract from a medical paper to make it more accessible and understandable to non-expert audiences, commonly referred to as "lay summaries". These lay summaries aim to present the key information from the original articles in a way that is less technical and contains more background information, making it easier for a broader range of readers,including researchers, medical professionals, journalists, and the general public, to comprehend the content.
    """
    completion = client.completions.create(
        model="BioMistral/BioMistral-7B-DARE-AWQ-QGS128-W4-GEMM",
        prompt=q_prompt + "\nAbstract: " + text + "\n Answer: ",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
    )
    # Assuming the response contains a field 'choices' with the rephrased text
    rephrased_text = completion.choices[0].text.strip()
    return rephrased_text


if __name__ == "__main__":
    DATA_FOLDER = "/data/colx531/biolaysumm2024_data/"
    file_names = ["PLOS_test.jsonl", "eLife_test.jsonl"]

    for file_name in file_names:
        with open(DATA_FOLDER + file_name, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for item in data:
            sections = item["article"].split("\n")
            item["sections"] = dict(zip(item["headings"], sections))

        # Modify OpenAI's API key and API base to use vLLM's API server.
        OPENAI_API_KEY = "EMPTY"
        OPENAI_API_BASE = "http://localhost:8000/v1"
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
        )

        predictions = []
        for item in tqdm(data, leave=True):
            # Extract abstracts
            abstract = item["sections"]["Abstract"]
            # Rephrase abstracts
            rephrased_abstract = rephrase_texts(abstract)
            predictions.append({"id": item["id"], "prediction": rephrased_abstract})

        with jsonlines.open(
            f'prediction_baseline2_BioMistral7B_{file_name.split(".", maxsplit=1)[0]}.jsonl',
            mode="w",
        ) as writer:
            writer.write_all(predictions)
