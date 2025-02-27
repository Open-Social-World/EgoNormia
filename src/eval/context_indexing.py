import json
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import os
import re
import requests
import base64
import random


from datasets import load_dataset
from google import genai
from google.genai import types
import concurrent.futures
import api_keys
import openai

class ImageIndexer:
    def __init__(self, embeddings=None, embeddings_path=None, test_time=False):

        # Load api key from environment variable
        self.model = genai.Client(api_key=api_keys.gem_key)

        # Build prompt
        self.build_prompt()

        # Convert to dict
        self.device = "cpu"

        self.index = None
        self.embeddings = None
        self.image_paths = []

        self.tt = test_time

        # Load embeddings and image paths if provided
        if embeddings is not None:
            self.embeddings = embeddings
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        elif embeddings_path:
            self.embeddings = np.load(embeddings_path)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)

        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # # Load data
        # with open('src/final_dataset/f_eleven_samples.json') as f:
        #     self.data = json.load(f)
        # Combine with other data
        with open(current_dir+'/final_dataset/final_data.json') as f:
            self.data = json.load(f)

        self.ds = load_dataset("open-social-world/EgoNormia")
        self.img_url = "https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/main/video/{img_id}/frame_all_prev.jpg?download=true"

        self.image_paths = [self.img_url.format(img_id=k) for k in self.data.keys()]

        # Load openai client
        self.oaiclient = openai.Client()
    
    @staticmethod
    def parse_response(response):
        pattern = r'\{(.*)\}'
        m = re.search(pattern, response, re.DOTALL)[0]
        return json.loads(m)

    def get_image_embedding(self, image_url):
        #"""Converts an image to an embedding using Gemini and a pre-trained model."""

        try:
            image = base64.b64encode(requests.get(image_url).content).decode('utf-8')

            image_file = types.Part.from_bytes(data=image,mime_type="image/jpeg")
            full_input = [self.description_prompt, image_file]

            response = self.model.models.generate_content(model = "gemini-1.5-flash-002",
                                                          contents = full_input
            )

            #text = self.parse_response(response.text)
            text = response.text

            embedding = self.parse_embedding(text)

            return embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize embeddings
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def parse_response(response):
        pattern = r'\{(.*)\}'
        m = re.search(pattern, response, re.DOTALL)[0]
        return json.loads(m)
        
    def parse_embedding(self, text):
        text = str(text)
        embedding = self.oaiclient.embeddings.create(
        input=text, model="text-embedding-3-small"
        ).data[0].embedding

        # Convert embeddings to tensor on cpu
        embedding = torch.tensor(embedding).to(self.device)
        embedding = torch.unsqueeze(embedding, 0)

        return embedding

    def run_indexing(self, image_paths):
        """Generates embeddings for a list of images and builds FAISS index."""

        print("Generating embeddings for images...")

        embeddings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            embeddings = list(tqdm(executor.map(self.get_image_embedding, image_paths), total=len(image_paths)))

        # Purge None embeddings, get indices of None embeddings
        self.none_indices = [i for i, e in enumerate(embeddings) if e is None]
        # Remove None embeddings (if in none_indices)
        embeddings = [e for i, e in enumerate(embeddings) if i not in self.none_indices]
        # Remove image_paths (if in none_indices)
        self.embeddings = np.vstack(embeddings).astype("float32")
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def save_index(self, embeddings_path="embeddings.npy"):
        """Saves the embeddings."""
        np.save(embeddings_path, self.embeddings)

    def query_image(self, query_image_path, top_k=5):

        """Finds the top-K similar images."""
        query_embedding = self.get_image_embedding(query_image_path).cpu().numpy()
        len_to_search = self.index.ntotal//2
        distances, indices = self.index.search(query_embedding, top_k)

        cr = []
        cj = []

        for i in range(0, 5):
            image_id = self.image_paths[indices[0][i]].split('video/')[-1].split('/')[0]
            datapoint = self.data[image_id]
            cr.append(datapoint['behaviors'][datapoint['correct']])
            cj.append(datapoint['justifications'][datapoint['correct']])

        #best_image_id = self.image_paths[indices[0][1]].split('prev_')[-1].split('.')[0]

        return cr

    def build_prompt(self):
        self.description_prompt = """
Your task is to analyze a first-person video of a person (the subject) performing an action given as a sequence of frames, and parse the activity of the video
as it relates to normative social behavior.

The subject is the person recording the video.

### Parse the following elements of the context:
1. Subject role: The job and intentions of the subject in the situation, i.e. waiter, customer, teacher, student, doctor, patient, etc. Include relevant attributes such as age, conditions, physical characteristics, etc.
2. Activity: The activity that the subject is doing.
3. Other people: Who else is present in the scene and what are they doing? Make sure this is distinct from the subject.

### Guidelines:
The output should be a single sentence describing the subject's role and the activity.
Exclude any information about the format of the video or clips itself.
AVOID flowery or emotional language, focus on concrete details.
Avoid any references to robotics or research - assume the situation (if being acted out) is real.


### Example:
If the situation is of the subject (a man) walking his large dog in a park where other children are playing, the output would be:

Man walking dog large dog on leash in public park with children playing.
"""
if __name__ == '__main__':
    #Load mappings to activities
    with open("../extras/id2high.json") as f:
        activities = json.load(f)   

    ac_nums = {}

    # Get numbers per activity
    for k in activities.keys():
        if activities[k] not in ac_nums:
            ac_nums[activities[k]] = [0, 0]
        else:
            ac_nums[activities[k]][0] += 1


    indexer = ImageIndexer()

    ids = indexer.ds['train']['id']

    # Build list of image urls from ds['train'] and img_url
    image_paths = [indexer.img_url.format(img_id=row) for row in ids]

    random.shuffle(image_paths)

    smaller_imgs = []

    fiftyfifty = True
    do_all = True

    if do_all:
        image_paths = image_paths
    elif fiftyfifty:
        # Pick 50% of samples per activity
        for img in image_paths:
            try:
                img_id = img.split('prev_')[-1].split('.')[0]
                
                act = activities[img_id]

                if ac_nums[act][1] > ac_nums[act][0]//2:
                    continue
                else:
                    ac_nums[act][1] += 1
                    smaller_imgs.append(img)
            except:
                f"Error with {img_id}"
    else:
        # Pick largest activity cluster, sample over all activities not in cluster
        largest = max(ac_nums, key=lambda x: ac_nums[x][0])

        for img in image_paths:
            try:
                img_id = img.split('prev_')[-1].split('.')[0]
                act = activities[img_id]

                if act != largest:
                    smaller_imgs.append(img)

            except:
                f"Error with {img_id}"


    # # Index images if "embeddings.npy" and "image_paths.json" do not exist
    if not os.path.exists("../normthinker/embeddings.npy"):
        print("Indexing images...")
        indexer.run_indexing(image_paths)
        indexer.save_index("../normthinker/embeddings.npy")
