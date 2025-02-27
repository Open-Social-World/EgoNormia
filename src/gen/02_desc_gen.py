###########################

# PREREQUISITES:
# data_samples.json: File containing snippets to be annotated, from previous steps

###########################

from google import genai
from google.genai import types
import json
from tqdm import tqdm
import requests
import base64
import os

import concurrent.futures

from datasets import load_dataset
import api_keys

import gen.prompts as ep


fast_model_name = "gemini-1.5-flash-002"

model = genai.Client(api_key=api_keys.gem_key)


source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f"{source_dir}/final_dataset/custom_data.json", "r") as f:
    data = json.loads(f.read())

annotated_ids = []

# Populate annotated_ids with keys of annotated data
for k in data.keys():
    if data[k]['desc'] != "":
        annotated_ids.append(data[k]['id'])

# Load the huggingface dataset
ds = load_dataset("open-social-world/EgoNormia")
img_url = "https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/main/video/{img_id}/frame_all_prev.jpg?download=true"

ids = ds['train']['id']

frame_urls = {i:img_url.format(img_id=i) for i in ids}

def process_sample(sample):
    video_id = sample['id']
    print(video_id)

    corresponding_frame = frame_urls[video_id]

    desc_prompt = ep.description_prompt
    image_bytes = base64.b64encode(requests.get(corresponding_frame).content).decode('utf-8')

    image_file = types.Part.from_bytes(data=image_bytes,mime_type="image/jpeg")
    full_input = [image_file, desc_prompt]
        

    response = model.models.generate_content(model = fast_model_name,
                                                contents = full_input
    )

    context = response.text
    print(context)
    print("\n")
    print('\n')

    print(context)

    return [context, video_id]

datalist = []
for k in data.keys():
    if k in annotated_ids:
        continue
    datapoint = data[k]
    datalist.append(datapoint)

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    desc_output = list(tqdm(executor.map(process_sample, datalist), total=len(datalist)))

descriptions = {}

for r in desc_output:
    if r is not None:
        descriptions[r[1]] = r[0]

# Iterate over rows in data, updating the 'desc' field
for k in data.keys():
    datapoint = data[k]
    if datapoint['id'] in descriptions:
        data[k]['desc'] = descriptions[datapoint['id']]

with open(f"{source_dir}/final_dataset/custom_data.json", "w") as f:
    json.dump(data, f, indent=4)
