import jsonlines
import json
from tqdm import tqdm


def baseline_model(item):
    return item['sections']['Abstract']


if __name__ == '__main__':

    data_folder = '/data/colx531/biolaysumm2024_data/'
    file_names = ['PLOS_test.jsonl', 'eLife_test.jsonl']

    for file_name in file_names:
        with open(data_folder+file_name, 'r') as f:
            data = [json.loads(line) for line in f]

        for item in data:
            sections = item['article'].split('\n')
            item['sections'] = {k: v for k, v in zip(item['headings'], sections)}

        predictions = []
        for item in tqdm(data, leave=True):
            rephrased_abstract = baseline_model(item)
            predictions.append({'id': item['id'], 'prediction': rephrased_abstract})

        with jsonlines.open(f'prediction_baseline1_Abstract_{file_name.split(".")[0]}.jsonl', mode='w') as writer:
            writer.write_all(predictions)
            
