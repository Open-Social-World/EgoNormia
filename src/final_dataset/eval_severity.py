import json
import time
import os
import tqdm
from datasets import load_dataset
import pickle

import api_keys
from eval.utils import backoff

from google import genai
from google.genai import types
import re
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import concurrent.futures

class SeverityAPI:
    def __init__(self, model, jsonfile, num_workers):

        self.modelname = model
        self.model = self.set_model()

        srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.jsonfile = srcdir+'/final_dataset/'+jsonfile
        savefile = self.jsonfile
        self.savefile = os.path.join(srcdir, '/final_dataset', savefile)

        self.num_workers = num_workers

        self.prefix = "The following descrption: {desc} describes a first-person perspective video of a person in a given situation"

    def set_model(self):
        model = genai.Client(api_key=api_keys.gem_key)

        return model
    
    def load_data_final(self):

        ds = load_dataset("open-social-world/EgoNormia")
        img_url = "https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/main/video/{img_id}/frame_all_prev.jpg?download=true"

        # Get target_vid_ids as ids of ds['train']
        target_vid_ids = ds['train']['id']

        # Check already-evaled rows
        eval = self.savefile

        with open(eval, 'r') as f:
            eval_results = json.load(f)
        
        task_set = []

        # Directly index columns of ds['train']
        behaviors_col = ds['train']['behaviors']
        justifications_col = ds['train']['justifications']
        correct_col = ds['train']['correct_idx']
        sensible_col = ds['train']['sensible_idx']
        desc_col = ds['train']['description']

        # For each id in target_vid_ids (recall id is in form uuid_timestamp)
        for cnt, vid_id in tqdm.tqdm(enumerate(target_vid_ids), desc="Loading data"):
            evl_res = eval_results[vid_id]

            # If data['answers'] has a key equal to self.modelname, skip
            if self.modelname in evl_res.keys():
                print(f"Skipping {vid_id}, already tested on {self.modelname}.")
                continue

            behaviors = behaviors_col[cnt]
            justifications = justifications_col[cnt]

            index_of_corr = correct_col[cnt]
            sensible = sensible_col[cnt] # These are indices


            correct_behavior = index_of_corr
            prev_images_paths = img_url.format(img_id=vid_id) # Single image

            # Build random mappings as current index:original index

            desc = desc_col[cnt]
            # Construct datapoint
            datapoint = {'id': vid_id,
                        'behaviors': behaviors,
                        'justifications': justifications,
                        'correct': [correct_behavior],
                        'sensible': sensible,
                        '_prev': prev_images_paths,
                        'description': desc}
            
            task_set.append(datapoint)

        print(f"Task set size: {len(task_set)}")

        task_set = task_set[:10]

        return task_set
    
    def assign_severity(self, datapoint):

        behaviors = datapoint['behaviors']
        correct = datapoint['correct'][0]

        correct_behavior = behaviors[correct]

        prefix = self.prefix.format(desc=datapoint['description'])

        prompt = prefix+f""" a person in a given situation.

The correct action to take in this situation is to {correct_behavior}.

Given the below list of violation levels, choose the closest level violation that takes place when this given correct action is NOT performed, 
i.e. when the person does NOT {correct_behavior}.

You shouldn't use the info in options to learn about the context, but rather to make a decision based on the severity of the violation.

Structure your answer as a single integer, where the integer corresponds to the index of the violation level in the list below, demarcated by the <VIOLATION> tag.

Think carefully about your response and reason over it in the output.

The only possible integers you should output are 1, 2, 3, 4, or 5.

Violation levels:
    1: "Minor Breach (1): This is a slightly inappropriate action that is mostly harmless, such as talking loudly on the phone in a public place. The social consequence is that it will likely be ignored or quickly forgiven, with perhaps just a glance or brief comment from others.",
    2: "Mild Violation (2): A mild violation is noticeable but not disruptive, like cutting in line without realizing it. The social consequence would be mild disapproval or a subtle correction, such as a polite reminder or light reprimand.",
    3: "Moderate Breach (3): A moderate breach clearly violates social norms and causes discomfort, such as interrupting someone in a formal meeting. The social consequence would be verbal correction or disapproving looks, with a firm reprimand or social distancing as a likely sanction.",
    4: "Severe Violation (4): A severe violation significantly disrupts social harmony, like making an offensive joke in public. This type of breach would result in strong disapproval and may harm relationships, with verbal confrontation or exclusion from the group as possible sanctions.",
    5: "Extreme Violation (5): An extreme violation is a major breach of social norms, such as stealing in a workplace or social gathering. The social consequence would be severe, possibly leading to formal punishment or permanent exclusion from the group, along with potential social isolation."

Response example:
<reasoning goes here>

<VIOLATION>1</VIOLATION>
"""

        a_results_text = self.inference(prompt) # Expect output in form of [0, 1]

        # Find integer bounded by <VIOLATION> tags
        a_results_text = re.findall(r'<VIOLATION>(.*?)</VIOLATION>', a_results_text)

        # Cast to int
        a_results_int = int(a_results_text[0])

        return [a_results_int, datapoint['id']]

    @backoff(max_retries=5, base_delay=3)
    def inference(self, prompt):

        full_input = [prompt]

        response = self.model.models.generate_content(model = self.modelname,
                                                      contents = full_input)

        return response.text
    
    def save_results(self, eval_results):

        already_sampled = {}

        with open(self.savefile, 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            if 'severity' in value.keys():
                already_sampled[key] = value

        # For each row in data, replace the scores
        for key, value in eval_results.items():
            if key in already_sampled.keys():
                continue
            else:
                data[key]['severity'] = value

        with open(self.savefile, 'w') as f:
            json.dump(data, f, indent=4)

    def evaluate(self):

        test_set = self.load_data_final()

        # Iterate over the test set
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            severity_futures = list(tqdm.tqdm(executor.map(self.assign_severity, test_set), total=len(test_set)))

        severity_temp = {k: v for v,k in severity_futures}

        eval_results = {}

        # Iterate over ids of test_set
        for dp in test_set:
            task_id = dp['id']

            eval_results[task_id] = severity_temp[task_id]

        # Once all samples are evaluated, compile results separately
        self.save_results(eval_results)

        print("Evaluation complete.")


if __name__ == "__main__":
    # Load data
    model = "gemini-1.5-flash-002"
    jsonfile = "final_data.json"
    num_workers = 1
    desc = "Eval severity"

    eval_api = SeverityAPI(model, jsonfile, num_workers)
    task_set = eval_api.load_data_final()

    eval_api.evaluate()