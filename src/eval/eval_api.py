import json
import time
import random
import os
import tqdm
from datasets import load_dataset
import requests
import base64
from anthropic import AnthropicVertex
import PIL.Image
import io
import pickle

import api_keys
from eval.utils import backoff, setup_logger, ReasoningCache

# Gemini imports
from google import genai
from google.genai import types
import re
import ast
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import openai
import concurrent.futures
from openai import AzureOpenAI

client = storage.Client()
bucket_name = 'physical-social-norm'
bucket = client.get_bucket(bucket_name)

class EvalAPI:
    def __init__(self, model, blind, jsonfile, num_workers, desc, num_datapoints):

        self.modelname = model
        self.model = self.set_model()

        self.blind = blind
        srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.jsonfile = srcdir+'/final_dataset/'+jsonfile
        savefile = self.jsonfile.replace('.json','_eval.json')
        self.savefile = os.path.join(srcdir, '/final_dataset', savefile)

        self.only_best = False

        self.num_workers = num_workers
        self.num_datapoints = num_datapoints

        self.custom = False

        self.desc = desc

        self.ablation = False

        self.logdir = f"{self.modelname}_{time.strftime('%Y%m%d_%H%M%S')}"

        self.logger = setup_logger(log_name=self.logdir)
        self.rc = ReasoningCache(dir_name=self.logdir)

        self.logger.info(f"Testing Conditions \n Model: {self.modelname} \n Blind: {self.blind} \n JSON File: {self.jsonfile}")
        self.logger.info(f"Desc: {self.desc}")
        self.logger.info(f"Only best: {self.only_best}")

        if self.only_best:
            self.logger.info("Only best mode enabled. Sensible and follow_norm tasks will not be evaluated.")
            time.sleep(1)

        if self.desc:
            self.prefix = "The following descrption: {desc} describes a first-person perspective video of a person in a given situation"
        elif not self.blind:
            self.prefix = "The following images from a first-person perspective video depict"
        else:
            self.prefix = "You are blind, so do not request context, only follow the instructions below. This situation involves"

        if self.blind:
            self.logger.info("Blind mode enabled. No images will be passed to the model.")
            self.modelname = "blind_" + self.modelname

        if 'rag' in self.modelname:
            from eval.context_indexing import ImageIndexer
            self.logger.info("Loading RAG model")

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
                self.logger.info(f"Skipping {vid_id}, already tested on {self.modelname}.")
                continue

            behaviors = behaviors_col[cnt]
            justifications = justifications_col[cnt]

            index_of_corr = correct_col[cnt]
            sensible = sensible_col[cnt] # These are indices

            n = len(behaviors)
            random_indices_behaviors = random.sample(random_indices_behaviors, n)
            random_indices_justifications = random.sample(random_indices_justifications, n)

            behaviors = [behaviors[i] for i in random_indices_behaviors]
            justifications = [justifications[i] for i in random_indices_justifications]
            sensible = [random_indices_behaviors[i] for i in sensible]


            correct_behavior = random_indices_behaviors[index_of_corr]
            correct_justification = random_indices_justifications[index_of_corr]

            if self.ablation != 'video' and self.ablation != 'discrete_frames':
                prev_images_paths = img_url.format(img_id=vid_id) # Single image
            elif self.ablation == 'discrete_frames':
                if vid_id in self.uploaded_videos1:
                    prev_images_paths = [f"https://storage.googleapis.com/physical-social-norm/sampled_frames_new_new/{vid_id}/frame_{i}_prev.jpg" for i in range(5)]
                elif vid_id in self.uploaded_videos2:
                    prev_images_paths = [f"https://storage.googleapis.com/physical-social-norm/sampled_frames_v2/{vid_id}/frame_{i}_prev.jpg" for i in range(5)]
            elif self.ablation == 'video':
                vid_url = "https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/main/video/{vid_id}/video_prev.mp4?download=true"
                prev_images_paths = vid_url.format(vid_id=vid_id)

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

        self.logger.info(f"Task set size: {len(task_set)}")

        if self.num_datapoints != -1:
            if self.num_datapoints > len(task_set):
                self.logger.info(f"Requested {self.num_datapoints} datapoints, but only {len(task_set)} available. Using all available datapoints.")
            else:
                self.logger.info(f"Requested {self.num_datapoints} datapoints. Using only the first {self.num_datapoints} datapoints.")
                task_set = task_set[:self.num_datapoints]

        # Make results directory if it doesn't exist
        if not os.path.exists(f'../results/{self.logdir}'):
            os.makedirs(f'../results/{self.logdir}')

        with open(f'../results/{self.logdir}/data.pkl', 'wb') as f:
            pickle.dump(task_set, f)

        return task_set      
    
    def pick_best(self, datapoint):

        behaviors = datapoint['behaviors']
        justifications = datapoint['justifications']
        correct = datapoint['correct']
        _prev = datapoint['_prev']

        if self.desc:
            prefix = self.prefix.format(desc=datapoint['description'])
        else:
            prefix = self.prefix
        prompt = prefix+f""" a person in a given situation.

Given the below list of behaviors, choose the single most normatively relevant or appropriate action to perform next.
You shouldn't use the info in options to learn about the context, but rather to make a decision based on the normative appropriateness of the behavior.

You shouldn't eliminate any options only based on the presence of elements in the context; you should focus on normative appropriateness.

Structure your answer as a single integer, where the integer corresponds to the index of the behavior in the list below.

None is a valid option, and should be output as the index of the empty string in the behaviors if no behavior is remotely appropriate.
Don't use None to indicate that you don't know the answer, only use it if you are certain that no behavior is appropriate.

Think carefully about your response and reason over it in the output.

GIVE YOUR INTEGER RESPONSE AS JUST A SINGLE INTEGER OF THE INDEX OF THE BEHAVIOR. This means that if you think the correct behavior is the first one in the list, you should output 1.

The only possible integers you should output are 1, 2, 3, 4, or 5.

Behavior options: {behaviors}

Response example:
<reasoning goes here>
1
"""
        justification_prompt = """ a person in a given situation.

"{behavior}" is selected as the most normatively relevant or appropriate action for the person to perform in the given situation.
However, this behavior might not be the most normatively correct action to perform in the given situation. Be open to the possibility that the behavior might be incorrect.

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

            a_results_text = self.inference(prompt, _prev) # Expect output in form of [2, 3]

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

            # Get unshuffler as inverse of datapoint['behavior_shuffle'] and datapoint['justification_shuffle']
            unshuffler_a = {v: k for k, v in datapoint['behavior_shuffle'].items()}
            unshuffler_j = {v: k for k, v in datapoint['justification_shuffle'].items()}

            if results[0] <= 4 and results[0] > -1:
                results[0] = unshuffler_a[results[0]]

            if results[1] <= 4 and results[1] > -1:
                results[1] = unshuffler_j[results[1]]

            full_results = [{'results': results, 'correct': correct}, datapoint['id']]

            self.logger.debug(f"{full_results}")

            return full_results
        except Exception as e:
            self.logger.warning(f"Error: {e}, skipping.")
            return [{'results': [], 'correct': correct}, datapoint['id']]

    def pick_sensible(self, datapoint):

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

            unshuffler_a = {v: k for k, v in datapoint['behavior_shuffle'].items()}

            if len(results) == len([r for r in results if r <= 4 and r > -1]):
                results = [unshuffler_a[r] for r in results]
            else:
                results = [] # Model outputs malform, but rest of the code is fine, counts as model error
            sensible = [datapoint['behavior_shuffle'][s] for s in sensible]

            full_results = [{'results': results, 'correct': sensible}, datapoint['id']]
            self.logger.debug(f"{full_results}")

            return full_results

        except Exception as e:
            self.logger.warning(f"Error: {e}, skipping.")
            return [{'results': [], 'correct': sensible}, datapoint['id']]

    def evaluate(self):

        test_set = self.load_data_final()

        # Iterate over the test set
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            best_futures = list(tqdm.tqdm(executor.map(self.pick_best, test_set), total=len(test_set)))

        if not self.only_best:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                sensible_futures = list(tqdm.tqdm(executor.map(self.pick_sensible, test_set), total=len(test_set)))

        # Make results directory if it doesn't exist
        if not os.path.exists(f'../results/{self.logdir}'):
            os.makedirs(f'../results/{self.logdir}')

        with open(f'../results/{self.logdir}/results.pkl', 'wb') as f:
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

            # Don't add point if malform i.e. skipped
            if best != [] and sensible != []:
                eval_results[task_id] = {'best': best, 'sensible': sensible, 'followed': follow}

        # Once all samples are evaluated, compile results separately
        self.save_results(eval_results)

        self.logger.info("Evaluation complete.")

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

class GeminiEvalAPI(EvalAPI):

    def __init__(self, model, blind, jsonfile, num_workers, desc, num_datapoints, ablation):
        # Super initialization
        super().__init__(model, blind, jsonfile, num_workers, desc, num_datapoints)

        # Initialize the ablation variable
        self.ablation = ablation

        if self.ablation != '':
            self.logger.info(f"Running ablation study of input types: {self.ablation}")

        if self.ablation == 'video' or self.ablation == 'discrete_frames':
            # Get list of already uploaded videos
            blobs = bucket.list_blobs()
            self.uploaded_videos1 = {i.name.split('/')[-1].split('_')[0] + '_' + i.name.split('/')[-1].split('_')[1] for i in blobs if i.name.startswith('sampled_snippets_new_new/') and len(i.name.split('/')[-1]) > 1}
            blobs = bucket.list_blobs()
            self.uploaded_videos2 = {i.name.split('/')[-1].split('_')[0] + '_' + i.name.split('/')[-1].split('_')[1] for i in blobs if i.name.startswith('sampled_snippets_v2/') and len(i.name.split('/')[-1]) > 1}
            blobs = bucket.list_blobs()

    def set_model(self):
        model = genai.Client(api_key=api_keys.gem_key)

        return model

    def inference(self, prompt, image):

        if self.blind:
            full_input = [prompt]
        else:
            if self.ablation != 'video' and self.ablation != 'discrete_frames':
                image_bytes = base64.b64encode(requests.get(image).content).decode('utf-8')

                image_file = types.Part.from_bytes(data=image_bytes,mime_type="image/jpeg")
                full_input = [prompt, image_file]
                mn = self.modelname.replace('blind_','').replace('desc_','')

            elif self.ablation == 'video':
                video_bytes = base64.b64encode(requests.get(image).content).decode('utf-8')
                video_file = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
                full_input = [prompt, video_file]
                mn = self.modelname.replace('video_','')

            elif self.ablation == 'discrete_frames':
                full_input = [prompt]
                for img in image:
                    image_bytes = base64.b64encode(requests.get(img).content).decode('utf-8')

                    img_file = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    full_input.append(img_file)

                mn = self.modelname.replace('frames_','')

        response = self.model.models.generate_content(model = mn,
                                                      contents = full_input
                                                                           
        )

        self.rc.add_and_write(self.modelname, prompt, response.text)

        return response.text
    
class OpenAIEvalAPI(EvalAPI):

    def set_model(self):

        # model = openai.Client()

        # return model
        endpoint = api_keys.azure_endpoint

        subscription_key = api_keys.azure_key
        api_version = "2024-12-01-preview"

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

        return client
    
    @backoff(max_retries=5, base_delay=3)
    def inference(self, prompt, image):

        contents = []
        if not self.blind:
            contents.append({"type": "image_url", "image_url": {"url":image}})
        contents.append({"type": "text", "text": prompt})

        mn = self.modelname.replace('blind_','').replace('desc_','')

        response = self.model.chat.completions.create(
            model = "gpt-4o-240513-72635",
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

        self.rc.add_and_write(self.modelname, prompt, response)

        return response
    
class OpenAIO3EvalAPI(EvalAPI):

    def set_model(self):

        model = openai.Client()

        return model
    
    @backoff(max_retries=5, base_delay=3)
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

        self.rc.add_and_write(self.modelname, prompt, response)

        return response

class RagEval(EvalAPI):

    def set_model(self):

        self.oaiclient = openai.OpenAI()

        return None # Model not used here as implicitly defined

    @backoff(max_retries=5, base_delay=3)
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

        self.rc.add_and_write(self.modelname, prompt, response)

        return response

class ClaudeEvalAPI(EvalAPI):
    
    def set_model(self):

        client = AnthropicVertex(region=api_keys.LOCATION, project_id=api_keys.PROJECT_ID)

        return client

    @backoff(max_retries=5, base_delay=3)
    def inference(self, prompt, image):
            contents = []

            # Load image and resize
            img = PIL.Image.open(io.BytesIO(requests.get(image).content))

            # Reduce image size to 25% to not exceed size limit
            img = img.resize((int(img.width * 0.25), int(img.height * 0.25)))

            byte_io = io.BytesIO()
            img.save(byte_io, format='JPEG')
            byte_data = byte_io.getvalue() 

            # Encode image to base64 after resizing
            image_b64 = base64.b64encode(byte_data).decode('utf-8')

            contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                }
            })
            contents.append({"type": "text", "text": prompt})

            temp_modelname = self.modelname.strip('blind_').strip('desc_')
            temp_modelname = "c" + temp_modelname

            response = self.model.messages.create(
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": contents
                    }
                ],

                model=temp_modelname,
                temperature=0.0,

            )

            list_response = response.content[0].text

            self.rc.add_and_write(self.modelname, prompt, list_response)
        
            return list_response