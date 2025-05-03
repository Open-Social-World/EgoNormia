import json
import os

target_model = 'gemini-2.5-flash-preview-04-17'

failures = []

script_dir = os.path.dirname(os.path.abspath(__file__))
final_dataset_dir = os.path.join(script_dir, '../final_dataset/final_data_eval.json')

with open(final_dataset_dir, 'r') as f:
    data = json.load(f)


for k,v in data.items():
    if target_model in v:
        if v[target_model]['best']['results'] != v[target_model]['best']['correct']:
            failures.append(k)


print(f"Number of failures: {len(failures)}")
