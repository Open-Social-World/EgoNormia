import api_keys
import json
from datasets import load_dataset
import tqdm
import random
import cv2
import sys

# # OpenAI imports
import openai

from eval.eval_api import EvalAPI, RateLimiterObject

# Define your custom eval here (i.e. modelname and custom inference pipeline)

class CustomEvalAPI(EvalAPI):
    def set_model(self):

        model = openai.Client()

        # Define self.modelname explicitly
        self.modelname = ""

        # Hardcode the rate limit for the model (in (calls-1)/min), as it is only known to you
        ratelimiter = RateLimiterObject()

        if self.modelname == "":
            raise ValueError("Please specify a model when running the custom eval API")
        
        self.custom = True

        if self.modelname == "":
            raise ValueError("Please specify a model when running the custom eval API")
            sys.exit(1)

        return model, ratelimiter

    def inference(self, prompt, image):
        # Modify this function as you see fit, so long as you are using only the data available per datapoint

        # Requirements:
        # Output should be a string
        # Output should contain either an integer or a list (in string format) at the end of the string

        # Load image from path

        img = cv2.imread(image)

        contents = []
        contents.append({"type": "image/jpg", "data": img})
        contents.append({"type": "text", "text": prompt})

        response = self.model.chat.completions.create(
            model = self.modelname,
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
    
    def load_data_custom(self):

        ds = load_dataset("open-social-world/EgoNormia")

        image_path = '../custom_video/frames/{img_id}.jpg'

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
            prev_images_paths = image_path.format(img_id=vid_id)

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