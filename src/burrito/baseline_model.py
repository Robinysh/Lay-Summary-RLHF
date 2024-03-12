import jsonlines
from icecream import ic
import json

with open('/data/colx531/biolaysumm2024_data/PLOS_val.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

for item in data:
    sections = item['article'].split('\n')
    item['sections'] = {k: v for k, v in zip(item['headings'], sections)}

def baseline_model(item):
    return item['sections']['Abstract']

predictions = []
for item in data:
    pred = baseline_model(item)
    predictions.append({'id': item['id'], 'prediction': pred})

with jsonlines.open('prediction.jsonl', mode='w') as writer:
    writer.write_all(predictions)