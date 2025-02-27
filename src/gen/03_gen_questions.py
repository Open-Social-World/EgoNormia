###########################

# PREREQUISITES:
# data_samples.json: File containing snippets to be annotated, from previous steps

###########################


from google import genai
from google.genai import types
import json
from tqdm import tqdm
import re
import os

import gen.prompts as ep
import concurrent.futures

from datasets import load_dataset
import api_keys
import openai

# Flags to set manually:
# 1. filepath: The path to the file containing the snippets to be annotated
# 2. parse_taxonomy: Set to True if you want to parse the taxonomy, False otherwise
# 3. fast_model_name: The name of the fast model to use for generation of the answers and justifications
# 4. tax_model_name: The name of the model to use for generation of the taxonomy and context

parse_taxonomy = True

fast_model_name = "gpt-4o"
oai_client = openai.Client()

tax_model_name = "gemini-1.5-flash-002"
tax_model = genai.Client(api_key=api_keys.gem_key)

source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(f"{source_dir}/final_dataset/custom_data.json", "r") as f:
    data = json.loads(f.read())

#### DEF ANNOTATION FILE HERE
filepath = f"{source_dir}/final_dataset/custom_data.json"
annotated_ids = set()
if os.path.exists(filepath):
    with open(f'{filepath}', 'r') as f:
        tempdat = json.loads(f.read())
        annotated_ids = [i for i in tempdat.keys() if tempdat[i]['justifications'][0] != '']

def parse_response(response):
    pattern = r'\{(.*)\}'
    m = re.search(pattern, response, re.DOTALL)[0]
    return json.loads(m)

# Load already-annotated datapoints
annot_keys = []

# Load the huggingface dataset
ds = load_dataset("open-social-world/EgoNormia")
img_url = "https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/main/video/{img_id}/frame_all_prev.jpg?download=true"

ids = ds['train']['id']

data = [data[i] for i in ids]

data = data[:3]

def process_sample(sample):
    id = sample['id']
    if id in annotated_ids:
        print(f'{id} already annotated')
        return None

    add_context = sample['desc']

    parsed_response = {}
    
    try:
        aj_prompt = ep.action_base_prompt.format(base=add_context)

        contents = []

        contents.append({"type": "text", "text": aj_prompt})

        response = oai_client.chat.completions.create(
            model = fast_model_name,
            messages=[
                {
                    "role": "user",
                    "content": contents
                }
            ],
            max_tokens=400,
            presence_penalty=1.8,
            temperature=0.5
        )

        act = parse_response(response.choices[0].message.content)

        print(act)
        print("\n")
        print('\n')
        sample['behaviors'] = act['Actions']
        sample['context'] = act['Contexts']

        parsed_response=sample.copy()

        return parsed_response
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Skipping...")

context_output = []

print("########################################")
print("Generating Actions")
print("########################################")

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    context_output = list(tqdm(executor.map(process_sample, data), total=len(data)))


with open(f"{source_dir}/final_dataset/custom_data.json", "r") as f:
    old_data = json.loads(f.read())

for row in context_output:
    if row is not None:
        row_id = row['id']
        old_data[row['id']] = row

with open(f'{filepath}', 'w') as f:
    json.dump(old_data, f, indent=4)

# Parsing taxonomy separately
if parse_taxonomy:

    print("########################################")
    print("Parsing Taxonomy and Justifications")
    print("########################################")

    def parse_tax_and_just(sample):

        try:

            behaviors = sample['behaviors']
            if 'context' in sample.keys():
                context = sample['context']
            else:
                context = ''

            tax_prompt = ep.taxonomy_prompt.format(pr = behaviors, td = ep.taxonomy_def)
            new_response = tax_model.models.generate_content(model = tax_model_name,
                                                        contents = [tax_prompt]
            )

            justification_prompt = ep.justification_prompt.format(context=context, action=behaviors)

            justification_response = tax_model.models.generate_content(model = tax_model_name,
                                                        contents = [justification_prompt]
            )
            taxonomy = parse_response(new_response.text)
            justification = parse_response(justification_response.text)

            sample['taxonomy'] = taxonomy
            sample['justifications'] = justification

            sample['correct'] = 0
            sample['sensibles'] = [0]

            new_data = sample.copy()

            return new_data
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Skipping...")

    if os.path.exists(filepath):
        with open(f'{filepath}', 'r') as f:
            tempdat = json.loads(f.read())

    not_sampled = []

    for i in tempdat.keys():
        if tempdat[i]["justifications"][0] == '':
            not_sampled.append(tempdat[i])

    not_sampled = not_sampled[:3]

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        tax_output = list(tqdm(executor.map(parse_tax_and_just, not_sampled), total=len(not_sampled)))

    with open(f"{source_dir}/final_dataset/custom_data.json", "r") as f:
        old_data = json.loads(f.read())

    for row in context_output:
        if row is not None:
            row_id = row['id']
            old_data[row['id']] = row
    with open(f'{filepath}', 'w') as f:
        json.dump(old_data, f, indent=4)