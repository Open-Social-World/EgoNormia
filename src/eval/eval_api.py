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
from transformers import AutoModelForImageTextToText, AutoTokenizer
import pickle
import threading

import api_keys
from eval.utils import backoff, setup_logger, ReasoningCache, InvalidResponseError, RefusalError, APIError

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

class EvalAPI:
    def __init__(self, model, blind, jsonfile, num_workers, desc, num_datapoints, split=None):

        self.modelname = model
        self.model = self.set_model()

        # Initialize threading event for stopping
        self.stop_event = threading.Event()

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

        if split is not None:
            splitfile = srcdir + '/final_dataset/' + split
            splitfile = os.path.join(srcdir, '/final_dataset', splitfile)
            with open(splitfile, 'r') as f:
                split_data = json.load(f)['split']
            self.split = split_data
        else:
            self.split = None

        self.logger.info(f"Testing Conditions \n Model: {self.modelname} \n Blind: {self.blind} \n JSON File: {self.jsonfile}")
        self.logger.info(f"Desc: {self.desc}")
        self.logger.info(f"Only best: {self.only_best}")
        if self.split is not None:
            self.logger.info(f"Split: {len(self.split)} items")

        if self.only_best:
            self.logger.info("Only best mode enabled. Sensible and follow_norm tasks will not be evaluated.")
            time.sleep(1)

        if self.desc:
            self.prefix = "The following description: ''' {desc} ''' describes a situation involving"
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

        # Load list of ids from final_data.json
        with open(self.jsonfile, 'r') as f:
            final_data = json.load(f)
            target_vid_ids = final_data.keys()

        img_url = "https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/main/video/{img_id}/frame_all_prev.jpg?download=true"

        # Check already-evaled rows
        eval = self.savefile

        with open(eval, 'r') as f:
            eval_results = json.load(f)
        
        task_set = []

        # Directly index each element of final_data based on target_vid_ids
        behaviors_col = []
        justifications_col = []
        correct_col = []
        sensible_col = []
        desc_col = []
        # Load dataset
        for vid_id in target_vid_ids:
            behaviors_col.append(final_data[vid_id]['behaviors'])
            justifications_col.append(final_data[vid_id]['justifications'])
            correct_col.append(final_data[vid_id]['correct'])
            sensible_col.append(final_data[vid_id]['sensibles'])
            desc_col.append(final_data[vid_id]['desc'])

        # For each id in target_vid_ids (recall id is in form uuid_timestamp)
        for cnt, vid_id in tqdm.tqdm(enumerate(target_vid_ids), desc="Loading data"):

            evl_res = eval_results[vid_id]

            # If data['answers'] has a key equal to self.modelname, skip
            if self.modelname in evl_res.keys():
                self.logger.info(f"Skipping {vid_id}, already tested on {self.modelname}.")
                continue

            # If split is not None, check if vid_id is in split
            if self.split is not None and vid_id not in self.split:
                self.logger.info(f"Skipping {vid_id}, not in split.")
                continue

            behaviors = behaviors_col[cnt]
            justifications = justifications_col[cnt]

            index_of_corr = correct_col[cnt]
            sensible = sensible_col[cnt] # These are indices

            n = len(behaviors)
            # random_indices_behaviors = random.sample(range(n), n)
            # random_indices_justifications = random.sample(range(n), n)
            random_indices_behaviors = [i for i in range(n)]
            random_indices_justifications = [i for i in range(n)]

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

        # Terminate if stop event is set
        if self.stop_event.is_set():
            self.logger.info("Stopping evaluation due to stop event.")
            return [{'results': [-2, -2], 'correct': datapoint['correct']}, datapoint['id']]

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
                prompt = prompt + f"\n IMPORTANT: The correct responses under the most similar contexts were '{cr}' However, these are only guidance, and must be interpreted generally in terms of meaning, you must be very careful as the context might still vary in some critical way."

            if self.blind or self.desc:
                _prev = None


            correct[0] = datapoint['behavior_shuffle'][correct[0]]
            correct[1] = datapoint['justification_shuffle'][correct[1]]

            a_results_text = self.inference(prompt, _prev) # Expect output in form of [2, 3]
            if a_results_text == None:
                raise InvalidResponseError("Model returned None for inference")

            # Find last integer in response and cast to int, call invalid response if not possible
            try:
                a_results = int(re.findall(r'\d+', a_results_text)[-1])
                a_results = a_results - 1
                bb = behaviors[a_results]
                assert a_results in [0, 1, 2, 3, 4] # Check if a_results is in range of behaviors
            except:
                raise InvalidResponseError("Model returned invalid response for 'action' subtask")

            just_p = self.prefix+justification_prompt.format(behavior = bb, justifications = justifications)
            j_results_text = self.inference(just_p, _prev)
            if j_results_text == None:
                raise InvalidResponseError("Model returned None for inference")
            
            try:
                j_results = int(re.findall(r'\d+', j_results_text)[-1])
                j_results = j_results - 1
                assert j_results in [0, 1, 2, 3, 4] # Check if j_results is in range of justifications
            except:
                raise InvalidResponseError("Model returned invalid response for 'justification' subtask")

            results = [a_results, j_results]

            # Get unshuffler as inverse of datapoint['behavior_shuffle'] and datapoint['justification_shuffle']
            unshuffler_a = {v: k for k, v in datapoint['behavior_shuffle'].items()}
            unshuffler_j = {v: k for k, v in datapoint['justification_shuffle'].items()}

            results[0] = unshuffler_a[results[0]]
            results[1] = unshuffler_j[results[1]]

            full_results = [{'results': results, 'correct': correct}, datapoint['id']]

            self.logger.debug(f"{full_results}")

            return full_results

        except InvalidResponseError as e:
            self.logger.error(f"Model refusal/malform, recording {e}. ID: {datapoint['id']}")

            return [{'results': [-1, -1], 'correct': correct}, datapoint['id']]
        except Exception as e:
            if 'GenerateRequestsPerDayPerProjectPerModel' in str(e):

                # If usage limit for day exceeded, set stop event to stop all tasks
                self.logger.error(f"Daily usage limit exceeded: {e}. Setting stop event for all tasks.")
                self.stop_event.set()
                return [{'results': [-2, -2], 'correct': correct}, datapoint['id']]

                
            else:
                self.logger.warning(f"Error: {e}, skipping. ID: {datapoint['id']}")
                return [{'results': [-2, -2], 'correct': correct}, datapoint['id']]

    def pick_sensible(self, datapoint):

        if self.stop_event.is_set():
            self.logger.info("Stopping evaluation due to stop event.")
            return [{'results': [-2, -2], 'correct': datapoint['sensible']}, datapoint['id']]

        behaviors = datapoint['behaviors']
        sensible = datapoint['sensible']
        _prev = datapoint['_prev']
        try:

            # Check if best_futures_temp for given id is [-2, -2], if so, throw ValueError, forcing a skip
            if datapoint['id'] in self.best_temp.keys():
                if self.best_temp[datapoint['id']]['results'] == [-2, -2]:
                    raise ValueError("Best futures temp for given id is [-2, -2], skipping.")

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
            if text_results == None:
                raise InvalidResponseError("Model returned None for inference")
            try:
                sensible_response = re.findall(r'\[.*\]', text_results)[-1]
                results = ast.literal_eval(sensible_response)
                results = [r - 1 for r in results]
            except:
                raise InvalidResponseError("Model returned invalid response for 'sensible' subtask")

            unshuffler_a = {v: k for k, v in datapoint['behavior_shuffle'].items()}

            if len(results) == len([r for r in results if r <= 4 and r > -1]):
                results = [unshuffler_a[r] for r in results]
            else:
                results = [] # Model outputs malform, but rest of the code is fine, counts as model error
            sensible = [datapoint['behavior_shuffle'][s] for s in sensible]

            full_results = [{'results': results, 'correct': sensible}, datapoint['id']]
            self.logger.debug(f"{full_results}")

            return full_results

        except InvalidResponseError as e:
            self.logger.error(f"Model refusal/malform, recording {e}. ID: {datapoint['id']}")

            return [{'results': [-1, -1], 'correct': sensible}, datapoint['id']]

        except Exception as e:
            if 'GenerateRequestsPerDayPerProjectPerModel' in str(e):

                # If usage limit for day exceeded, set stop event to stop all tasks
                self.logger.error(f"Daily usage limit exceeded: {e}. Setting stop event for all tasks.")
                self.stop_event.set()
                return [{'results': [-2, -2], 'correct': sensible}, datapoint['id']]

            else:
                self.logger.warning(f"Error: {e}, skipping.")
                return [{'results': [-2, -2], 'correct': sensible}, datapoint['id']]

    def evaluate(self):
        """
        Evaluates the model on the test set, with support for early stopping.
        """
        test_set = self.load_data_final()
        self.best_temp = {}
        self.stop_event.clear() # Clear the stop event for a new evaluation run

        #self.logger.info("Starting 'pick_best' evaluation with ThreadPoolExecutor.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks and store futures
            futures_best = {executor.submit(self.pick_best, dp): dp['id'] for dp in test_set}
            
            best_futures = []
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures_best), total=len(futures_best), desc="Evaluating 'pick_best'"):
                task_id = futures_best[future]
                try:
                    result = future.result()
                    best_futures.append(result)
                except concurrent.futures.CancelledError:
                    self.logger.info(f"Task for {task_id} was cancelled for 'pick_best'.")
                    best_futures.append([{'results': [-2, -2], 'correct': [None, None]}, task_id])
                except Exception as exc:
                    self.logger.error(f"'{task_id}' generated an exception during 'pick_best': {exc}")
                    best_futures.append([{'results': [-2, -2], 'correct': [None, None]}, task_id])

                # Check if the stop event is set to gracefully exit the loop
                if self.stop_event.is_set():
                    self.logger.warning("Stop event detected during 'pick_best' evaluation. Cancelling remaining tasks.")
                    for remaining_future in futures_best:
                        if not remaining_future.done():
                            remaining_future.cancel() # Attempt to cancel pending futures
                    break # Exit the as_completed loop

        self.best_temp = {k: v for v, k in best_futures}

        sensible_temp = {}
        sensible_futures = []

        if not self.only_best:
            # If stop event is set, skip sensible evaluation
            if not self.stop_event.is_set():
                #self.logger.info("Starting 'pick_sensible' evaluation with ThreadPoolExecutor.")
            
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures_sensible = {executor.submit(self.pick_sensible, dp): dp['id'] for dp in test_set}
                    
                    for future in tqdm.tqdm(concurrent.futures.as_completed(futures_sensible), total=len(futures_sensible), desc="Evaluating 'pick_sensible'"):
                        task_id = futures_sensible[future]
                        try:
                            result = future.result()
                            sensible_futures.append(result)
                        except concurrent.futures.CancelledError:
                            self.logger.info(f"Task for {task_id} was cancelled for 'pick_sensible'.")
                            sensible_futures.append([{'results': [-2, -2], 'correct': None}, task_id])
                        except Exception as exc:
                            self.logger.error(f"'{task_id}' generated an exception during 'pick_sensible': {exc}")
                            sensible_futures.append([{'results': [-2, -2], 'correct': None}, task_id])

                        # Check if the stop event is set to gracefully exit the loop
                        if self.stop_event.is_set():
                            self.logger.warning("Stop event detected during 'pick_sensible' evaluation. Cancelling remaining tasks.")
                            for remaining_future in futures_sensible:
                                if not remaining_future.done():
                                    remaining_future.cancel() # Attempt to cancel pending futures
                            break # Exit the as_completed loop
                sensible_temp = {k: v for v, k in sensible_futures}

        # Make results directory if it doesn't exist
        results_dir = f'../results/{self.logdir}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # It's better to save the raw futures results if you might need them for debugging later
        with open(os.path.join(results_dir, 'best_futures.pkl'), 'wb') as f:
            pickle.dump(best_futures, f)
        if not self.only_best:
            with open(os.path.join(results_dir, 'sensible_futures.pkl'), 'wb') as f:
                pickle.dump(sensible_futures, f)

        eval_results = {}

        # Iterate over ids of test_set
        for dp in test_set:
            task_id = dp['id']

            best = self.best_temp[task_id]
            sensible = sensible_temp[task_id] # Ensure sensible is populated for all tasks

            follow = {} # Not used in this snippet, but kept for context

            # Don't add point if skipped due to stop signal, add (none) response if invalid/refusal answer
            # Glossary:
            # [-1, -1] -> Invalid response (model refused or response malformed)
            # [-2, -2] -> API error (e.g., rate limit exceeded) 
            # [0-4, 0-4] -> Valid response, saved with results
            # Anything else -> Skipped, not saved

            if best['results'] != [-2, -2] and (s in [0, 1, 2, 3, 4] for s in sensible['results']):
                eval_results[task_id] = {'best': best, 'sensible': sensible, 'followed': follow}
            else:
                self.logger.info(f"Skipping {task_id} from final results due to incomplete/failed evaluation (best: {best['results']}, sensible: {sensible['results']}).")


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

    def __init__(self, model, blind, jsonfile, num_workers, desc, num_datapoints, ablation, split):
        # Super initialization
        super().__init__(model, blind, jsonfile, num_workers, desc, num_datapoints, split)

        # Initialize the ablation variable
        self.ablation = ablation

        if self.ablation != '':
            self.logger.info(f"Running ablation study of input types: {self.ablation}")

        if self.ablation == 'video' or self.ablation == 'discrete_frames':

            client = storage.Client()
            bucket_name = 'physical-social-norm'
            bucket = client.get_bucket(bucket_name)
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

        if self.blind or self.desc:
            full_input = [prompt]
            mn = self.modelname.replace('blind_','').replace('desc_','')
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
    
class AzureOpenAIEvalAPI(EvalAPI):

    def set_model(self):

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
        if not self.blind and not self.desc:
            contents.append({"type": "image_url", "image_url": {"url":image}})
        contents.append({"type": "text", "text": prompt})

        mn = self.modelname.replace('blind_','').replace('desc_','')

        if 'o4' in mn or 'o3' in mn:
            response = self.model.chat.completions.create(
                model = mn,
                reasoning_effort="medium",
                messages=[
                    {
                        "role": "user",
                        "content": contents
                    }
                ]
            )
        else:

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
    
class OpenAIEvalAPI(EvalAPI):

    def set_model(self):

        model = openai.Client()

        return model
    
    @backoff(max_retries=5, base_delay=3)
    def inference(self, prompt, image):

        contents = []
        if not self.blind and not self.desc:
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

        self.rc.add_and_write(self.modelname, prompt, response)

        return response

class RagEval(EvalAPI):

    def set_model(self):

        self.oaiclient = openai.OpenAI()

        return None # Model not used here as implicitly defined

    @backoff(max_retries=5, base_delay=3)
    def inference(self, prompt, image):

        contents = []
        if not self.blind and not self.desc:
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

            if not self.blind and not self.desc:
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


# class HuggingfaceEvalAPI(EvalAPI):

#     def set_model(self):

#         mn = self.modelname.replace('blind_','').replace('desc_','')
        
#         model = AutoModelForImageTextToText.from_pretrained(
#             mn,
#             torch_dtype="auto",
#             device_map="auto"
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

#         return model
    
#     def inference(self, prompt, image):

#         contents = []

#         contents.append({"type": "text", "text": prompt})

#         if not self.blind:
#             img = PIL.Image.open(io.BytesIO(requests.get(image).content))
#             contents.append({"type": "image", "image": img})

#         tokenized_content = self.tokenizer.apply_chat_template(contents,
#                                                                tokenize=False
#         )

#         inputs = self.tokenizer(tokenized_content, return_tensors="pt").to(self.model.device)

#         response = self.model.generate(**inputs)

#         response = self.tokenizer.decode(response[0], skip_special_tokens=True)[0]

#         return response

class VLLMEvalAPI(EvalAPI):

    def set_model(self):
        client = openai.OpenAI(
            api_key=api_keys.oai_key,
            base_url=api_keys.openai_api_base,
        )

        return client
    
    @backoff(max_retries=5, base_delay=3)
    def inference(self, prompt, image):

        contents = []
        if not self.blind and not self.desc:
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