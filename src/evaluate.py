from eval import eval_api
from eval import custom_eval_api
import argparse

parser = argparse.ArgumentParser(description="A simple command-line tool.")

parser.add_argument('--blind', action='store_true', help="Set blind mode")
parser.add_argument('--description', action='store_true', help="Show description")
parser.add_argument('--jsonfile', type=str, default='src/final_dataset/final_data.json', help="Path to the JSON file")
parser.add_argument('--modelname', type=str, default='', help="Name of the model")

args = parser.parse_args()

blind = args.blind
description = args.description
jsonfile = args.jsonfile
model_type = args.modelname

if model_type == '':
    print("Please specify a model type")
    exit()

if 'o3' in model_type:
    if not description:
        print("o3 models require description to be set to True")
        exit()

if description and blind:
    print("Description and blind are mutually exclusive")
    exit()

# Define which model you're using - make sure you select the right rate limit in calls/min
if 'o3' in model_type:
    model = eval_api.OpenAIO3EvalAPI(model=model_type, rl=60, blind=blind, jsonfile=jsonfile, num_workers=10, desc=description)
elif 'gemini' in model_type:
    model = eval_api.GeminiEvalAPI(model=model_type, rl=60, blind=blind, jsonfile=jsonfile, num_workers=10, desc=description)
elif 'gpt' in model_type:
    model = eval_api.OpenAIEvalAPI(model=model_type, rl=60, blind=blind, jsonfile=jsonfile, num_workers=10, desc=description)
elif 'rag' in model_type.lower():
    model = eval_api.RagEval(model=model_type, rl=60, blind=blind, jsonfile=jsonfile, num_workers=10, desc=description)
elif 'custom' in model_type:
    model = custom_eval_api.CustomEvalAPI(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=10, desc=description)

model.evaluate()


