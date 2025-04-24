from eval import eval_api
from eval import custom_eval_api
import argparse

parser = argparse.ArgumentParser(description="A simple command-line tool.")

parser.add_argument('--blind', action='store_true', help="Set blind mode")
parser.add_argument('--description', action='store_true', help="Show description")
parser.add_argument('--jsonfile', type=str, default='src/final_dataset/final_data.json', help="Path to the JSON file")
parser.add_argument('--modelname', type=str, default='', help="Name of the model")
parser.add_argument('--workers', type=int, default=1, help="Number of workers")
parser.add_argument('--number', type=int, default=-1, help="Number of datapoints to evaluate")
parser.add_argument('--ablation', type=str, default='', help="Ablation study of input types, can be 'video', 'discrete_frames' ")
parser.add_argument('--azure', action='store_true', help="Use Azure OpenAI API")

args = parser.parse_args()

blind = args.blind
description = args.description
jsonfile = args.jsonfile
model_type = args.modelname
num_workers = args.workers
num_datapoints = args.number
ablation = args.ablation
use_azure = args.azure

if model_type == '':
    print("Please specify a model type")
    exit()

if 'o3' in model_type or 'o4' in model_type and not description:
    print("o3/o4 models require description to be set to True")
    exit()

if description and blind:
    print("Description and blind are mutually exclusive")
    exit()

if ablation != '' and 'gemini' not in model_type:
    print("Ablation study of input types is currently only supported for Gemini models")
    exit()

if ablation != '' and blind:
    print("Ablation study of input types and blind are mutually exclusive - ablation study requires the model to be able to see the input")
    exit()

# Define which model you're using
if 'o3' in model_type:
    model = eval_api.OpenAIEvalAPI(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=num_workers, desc=description, num_datapoints=num_datapoints)
elif 'gemini' in model_type:
    model = eval_api.GeminiEvalAPI(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=num_workers, desc=description, num_datapoints=num_datapoints, ablation=ablation)
elif 'gpt' in model_type:
    if use_azure:
        model = eval_api.AzureOpenAIEvalAPI(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=num_workers, desc=description, num_datapoints=num_datapoints)
    else:
        model = eval_api.OpenAIEvalAPI(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=num_workers, desc=description, num_datapoints=num_datapoints)
elif 'rag' in model_type.lower():
    model = eval_api.RagEval(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=num_workers, desc=description, num_datapoints=num_datapoints)
elif 'claude' in model_type:
    model = eval_api.ClaudeEvalAPI(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=num_workers, desc=description, num_datapoints=num_datapoints)
elif 'custom' in model_type:
    model = custom_eval_api.CustomEvalAPI(model=model_type, blind=blind, jsonfile=jsonfile, num_workers=num_workers, desc=description, num_datapoints=num_datapoints)

model.evaluate()