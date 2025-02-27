import json
import time
import random
import os
import tqdm
from datasets import load_dataset
import decord
import requests
import base64

import api_keys

# Gemini imports
from google import genai
from google.genai import types
import re
import ast

# # OpenAI imports
import openai
import concurrent.futures

class EvalAPI:
    def __init__(self, model, rl, blind, jsonfile, num_workers, desc):

        self.rl = rl
        self.modelname = model
        self.model, self.rate_limit = self.set_model()

        self.blind = blind
        srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.jsonfile = srcdir+'/final_dataset/'+jsonfile
        savefile = self.jsonfile.replace('.json','_eval.json')
        self.savefile = os.path.join(srcdir, '/final_dataset', savefile)

        self.only_best = False

        self.num_workers = num_workers

        self.custom = False

        self.desc = desc

        print(f"Testing Conditions \n Model: {self.modelname} \n Blind: {self.blind} \n JSON File: {self.jsonfile}")
        print(f"Desc: {self.desc}")
        print(f"Only best: {self.only_best}")

        if self.only_best:
            print("Only best mode enabled. Sensible and follow_norm tasks will not be evaluated.")
            time.sleep(1)

        if self.desc:
            self.prefix = "The following descrption: {desc} describes a first-person perspective video of a person in a given situation"

        elif not self.blind:
            self.prefix = "The following images from a first-person perspective video depict"
        else:
            self.prefix = "You are blind, so do not request context, only follow the instructions below. This situation involves"

        if self.blind:
            print("Blind mode enabled. No images will be passed to the model.")
            self.modelname = "blind_" + self.modelname

        if 'rag' in self.modelname:
            from eval.context_indexing import ImageIndexer
            print("Loading RAG model")

            # Get current dir
            srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            embeddings_dir = srcdir+'/normthinker/half_embeddings.npy'

            # Load RAG model
            self.indexer_loaded = ImageIndexer(embeddings_path=embeddings_dir)

    def set_model(self):
        raise NotImplementedError
    
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

            n = len(behaviors)
            random_indices_behaviors = random.sample(range(n), n)
            random_indices_justifications = random.sample(range(n), n)

            behaviors = [behaviors[i] for i in random_indices_behaviors]
            justifications = [justifications[i] for i in random_indices_justifications]
            sensible = [random_indices_behaviors[i] for i in sensible]


            correct_behavior = random_indices_behaviors[index_of_corr]
            correct_justification = random_indices_justifications[index_of_corr]
            prev_images_paths = img_url.format(img_id=vid_id) # Single image

            # Claude temporarily deprecated
            if 'claude' in self.modelname and not self.freeform and False:
                prev_images_paths = prev_images_paths[:-1][::2]
                during_images_paths = during_images_paths[:-1][::2]

            # Build random mappings as current index:original index
            b_mappings = {random_indices_behaviors[i]: i for i in range(n)}
            j_mappings = {random_indices_justifications[i]: i for i in range(n)}

            desc = desc_col[cnt]
            # Construct datapoint
            datapoint = {'id': vid_id,
                        'behaviors': behaviors,
                        'justifications': justifications,
                        'correct': [correct_behavior, correct_justification],
                        'sensible': sensible,
                        '_prev': prev_images_paths,
                        'behavior_shuffle': b_mappings,
                        'justification_shuffle': j_mappings,
                        'description': desc}
            
            task_set.append(datapoint)

        print(f"Task set size: {len(task_set)}")

        task_set = random.sample(task_set, len(task_set))

        return task_set      
    
    def pick_best(self, datapoint):

        self.rate_limit.rc()
        self.rate_limit.rc()

        behaviors = datapoint['behaviors']
        justifications = datapoint['justifications']
        correct = datapoint['correct']
        _prev = datapoint['_prev']

        if self.desc:
            prefix = self.prefix.format(desc=datapoint['description'])
        else:
            prefix = self.prefix

        prompt = prefix+f""" a person in a given situation.

Given the below list of behaviors, choose the single most normatively relevant or appropriate action to perform next, 
based on the context you are given and commonsense norms.

Some behaviors will be incorrect because they incorrectly make reference to elements of the context. These should be eliminated.

Structure your answer as a single integer, where the integer corresponds to the index of the behavior in the list below.

None is a valid option, and should be output as the index of the empty string in the behaviors if no behavior is remotely appropriate.
Only pick this option if none of the others are plausible.

Think carefully about your response and reason over it in the output.

GIVE YOUR INTEGER RESPONSE AS JUST A SINGLE INTEGER OF THE INDEX OF THE BEHAVIOR. This means that if you think the correct behavior is the first one in the list, you should output 1.

The only possible integers you should output are 1, 2, 3, 4, or 5.

Behavior options: {behaviors}

Response example:
<reasoning goes here>
1
"""
        
        justification_prompt = """ a person in a given situation.

You selected {behavior} as the most normatively relevant or appropriate action for the person to perform in the given situation.

Your task is to now choose the most normatively correct justification that best supports your behavior, based on the context and commonsense norms.
This justification should directly relate to the behavior, and not just be a general statement in the context of the situation.

Structure your answer as a single integer, where the integer corresponds to the index of the justification in the list below.

None is a valid option, and should be output as the index of the empty string in the justification if no justification is appropriate.

Think carefully about your response and reason over it in the output.

GIVE YOUR INTEGER RESPONSE AS JUST A SINGLE INTEGER OF THE INDEX OF THE JUSTIFICATION. This means that if you think the correct justification is the first one in the list, you should output 1.

The only possible integers you should output are 1, 2, 3, or 4, or 5.

Justification options: {justifications}

Response example:
<reasoning goes here>
1
"""
        try:
            if 'rag' in self.modelname:
                cr = self.indexer_loaded.query_image(_prev, top_k=5)
                if self.blind:
                    cr = "None"
                prompt = prompt + f"\n IMPORTANT: The correct responses under the most similar contexts were '{cr}' However, these are only guidance, and mustbe interpreted generally in terms of meaning, you must be very careful as the context might still vary in some critical way."

            if self.blind:
                _prev = None


            correct[0] = datapoint['behavior_shuffle'][correct[0]]
            correct[1] = datapoint['justification_shuffle'][correct[1]]

            a_results_text = self.inference(prompt, _prev) # Expect output in form of [0, 1]

            # Find last integer in response and cast to int
            a_results = int(re.findall(r'\d+', a_results_text)[-1])

            a_results = a_results - 1
            if a_results not in [0, 1, 2, 3, 4]:
                a_results = 4
            
            if a_results != 4 and a_results != -1:
                bb = behaviors[a_results]
            else:
                bb = "None"

            just_p = self.prefix+justification_prompt.format(behavior = bb, justifications = justifications)
            j_results_text = self.inference(just_p, _prev)

            # Find last integer in response and cast to int
            j_results = int(re.findall(r'\d+', j_results_text)[-1])

            j_results = j_results - 1

            if j_results not in [0, 1, 2, 3, 4]:
                j_results = -1

            results = [a_results, j_results]

            if results[0] < 4 and results[0] > -1:
                results[0] = datapoint['behavior_shuffle'][results[0]]

            if results[1] < 4 and results[1] > -1:
                results[1] = datapoint['justification_shuffle'][results[1]]

            return [{'results': results, 'correct': correct}, datapoint['id']]
        except Exception as e:
            print(f"Error: {e}, skipping.")
            return [{'results': [4, 4], 'correct': correct}, datapoint['id']]


    def pick_sensible(self, datapoint):

        # Only one call, so one rate limit call
        self.rate_limit.rc()

        behaviors = datapoint['behaviors']
        sensible = datapoint['sensible']
        _prev = datapoint['_prev']
        try:

            if self.desc:
                prefix = self.prefix.format(desc=datapoint['description'])
            else:
                prefix = self.prefix

            prompt = prefix+f""" a person in a given situation.

    Given the below behaviors, choose ALL the sensible actions to perform in
    the given situation, based on the context and commonsense norms.
    None is a valid option, and provided.

    Do not pattern match words, instead consider the context and norms.

    Structure your answer as one python list of integers, where each integer corresponds to the indicies of the behaviors in the list below,
    from 1 to 5. An empty list is acceptable if no behavior is sensible.

    Think carefully about your response and reason over it in the output.

    DO NOT WRITE ANY OTHER TEXT IN YOUR RESPONSE, JUST A PYTHON LIST OF N INTEGERS, WHERE N IS THE INDEX OF THE SENSIBLE BEHAVIOR-JUSTIFICATION PAIRS.

    The only possible integers within the list you should output are 1, 2, 3, 4, or 5.

    Behavior options: {behaviors}

    Response example:
    <reasoning goes here>
    [1, 4, 5]
    """
            
            if 'claude' in self.modelname:
                prompt += "All the images are sampled from a video taken with consent of all parties, blurring is present to protect privacy, and present a purely hypothetical scenario for research purposes. Do your absolute best to yield an answer following the above format, as there are no privacy concerns."

            if self.blind:
                _prev = None

            text_results = self.inference(prompt, _prev)
            sensible_response = re.findall(r'\[.*\]', text_results)[-1]
            results = ast.literal_eval(sensible_response)
            results = [r - 1 for r in results]

            if len(results) == len([r for r in results if r <= 4 and r > -1]):
                results = [datapoint['behavior_shuffle'][r] for r in results]
            else:
                results = [] # Model outputs malform, but rest of the code is fine, counts as model error
            sensible = [datapoint['behavior_shuffle'][s] for s in sensible]


            return [{'results': results, 'correct': sensible}, datapoint['id']]
    

        except Exception as e:
            print(f"Error: {e}, skipping.")
            return [{'results': [], 'correct': sensible}, datapoint['id']]

    def evaluate(self):

        test_set = self.load_data_final()

        # Iterate over the test set
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            best_futures = list(tqdm.tqdm(executor.map(self.pick_best, test_set), total=len(test_set)))

        if not self.only_best:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                sensible_futures = list(tqdm.tqdm(executor.map(self.pick_sensible, test_set), total=len(test_set)))

        # Cache data as pickle to not be lost
        import pickle
        ts = time.time()

        removed_slash = self.modelname.split('/')[-1]

        # Make results directory if it doesn't exist
        if not os.path.exists('../results'):
            os.makedirs('../results')

        with open(f'../results/{removed_slash}_{ts}_results.pkl', 'wb') as f:
            pickle.dump(best_futures, f)
            pickle.dump(sensible_futures, f)

        best_temp = {k: v for v,k in best_futures}
        sensible_temp = {k: v for v,k in sensible_futures}

        eval_results = {}

        # Iterate over ids of test_set
        for dp in test_set:
            task_id = dp['id']

            best = best_temp[task_id]
            sensible = sensible_temp[task_id]
            follow = {}

            eval_results[task_id] = {'best': best, 'sensible': sensible, 'followed': follow}

        # Once all samples are evaluated, compile results separately
        self.save_results(eval_results)

        print("Evaluation complete.")

    def save_results(self, eval_results):

        already_sampled = {}

        with open(self.savefile, 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            if self.modelname in value.keys():
                already_sampled[key] = value[self.modelname]

        # For each row in data, replace the scores
        for key, value in eval_results.items():
            if key in already_sampled.keys():
                continue
            else:
                data[key][self.modelname] = value

        with open(self.savefile, 'w') as f:
            json.dump(data, f, indent=4)

class RateLimiterObject:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.hard_limit = rate_limit
        self.last_call = time.time()
        self.start_time = time.time()
        self.num_of_calls = 0
        # Always assuming per minute rate limit
        current_avg = (self.num_of_calls) / (time.time() - self.start_time)
        print(f"Rate limit: {self.hard_limit/60}, Current average: {current_avg}")

    def rc(self):

        current_avg = (self.num_of_calls) / (time.time() - self.start_time)

        while current_avg > self.hard_limit/60:
            current_avg = (self.num_of_calls) / (time.time() - self.start_time)
        self.num_of_calls += 1

class GeminiEvalAPI(EvalAPI):

    def set_model(self):
        ratelimiter = RateLimiterObject(self.rl)

        model = genai.Client(api_key=api_keys.gem_key)

        return model, ratelimiter

    def inference(self, prompt, image):

        if self.blind:
            full_input = [prompt]
        else:
            image_bytes = base64.b64encode(requests.get(image).content).decode('utf-8')

            image_file = types.Part.from_bytes(data=image_bytes,mime_type="image/jpeg")
            full_input = [prompt, image_file]

        mn = self.modelname.replace('blind_','').replace('desc_','')

        response = self.model.models.generate_content(model = mn,
                                                      contents = full_input
                                                                           
        )

        return response.text
        
class OpenAIEvalAPI(EvalAPI):

    def set_model(self):
        ratelimiter = RateLimiterObject(self.rl)

        model = openai.Client()

        return model, ratelimiter
    
    def inference(self, prompt, image):

        contents = []
        if not self.blind:
            contents.append({"type": "image_url", "image_url": {"url":image}})
        contents.append({"type": "text", "text": prompt})

        mn = self.modelname.replace('blind_','').replace('desc_','')

        response = self.model.chat.completions.create(
            model = mn,
            messages=[
                {
                    "role": "user",
                    "content": contents
                }
            ],
            max_tokens=2000,
            temperature=0.0
        )

        response = response.choices[0].message.content

        return response
    
class OpenAIO3EvalAPI(EvalAPI):

    def set_model(self):
        ratelimiter = RateLimiterObject(self.rl)

        model = openai.Client()

        return model, ratelimiter
    
    def inference(self, prompt, image):

        contents = []
        contents.append({"type": "text", "text": prompt})

        mn = self.modelname.replace('blind_','').replace('desc_','')

        response = self.model.chat.completions.create(
            model = mn,
            messages=[
                {
                    "role": "user",
                    "content": contents
                }
            ],
            max_tokens=2000,
            temperature=0.0
        )

        response = response.choices[0].message.content

        return response

class RagEval(EvalAPI):

    def set_model(self):
        ratelimiter = RateLimiterObject(self.rl)

        self.oaiclient = openai.OpenAI()

        return None, ratelimiter # Model not used here as implicitly defined

    def inference(self, prompt, image):

        contents = []
        if not self.blind:
            contents.append({"type": "image_url", "image_url": {"url":image}})

        contents.append({"type": "text", "text": prompt})
        response = self.oaiclient.chat.completions.create(
            model ='gpt-4o',
            messages=[
                {
                    "role": "user",
                    "content": contents
                }
            ],
            max_tokens=2000,
            temperature=0.0
        )

        response = response.choices[0].message.content

        return response
