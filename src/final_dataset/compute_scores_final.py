import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--noblind', type=str, required=False, help='Report blind')
parser.add_argument('--file', type=str, required=False, help='Target eval file', default='final_data')
parser.add_argument('--split', type=str, required=False, help='Split to use (Specify as json file path without .json extension)')
args = parser.parse_args()
if args.noblind:
    noblind = True
else:
    noblind = False
file = args.file
with open(f'{file}_eval.json', 'r') as f:
    data = json.load(f)

if args.split:
    with open(f'{args.split}.json', 'r') as f:
        split = json.load(f)['split']
    data = {k: v for k, v in data.items() if k in split}
else:
    split = None

scores = {}
missed_ids = {}
modeltotals = {}
nets = {}
for key in data:
    datapoint = data[key]
    answers = datapoint
    for model in answers:
        if noblind and 'blind' in model:
            continue
        nets[model] = 0
        scores[model] = {
                'best': {'both': 0, 'a': 0, 'j': 0},
                'sensible': 0
            }
        modeltotals[model] = 0
malforms = []
for key in data:
    datapoint = data[key]
    answers = datapoint
    for model in sorted(list(answers.keys())):
        try:
            if noblind and 'blind' in model:
                continue
            if model:
                # Best
                best_a = answers[model]['best']['results'][0] == answers[model]['best']['correct'][0]
                best_j = answers[model]['best']['results'][1] == answers[model]['best']['correct'][1]
                if best_a:
                    nets[model] += 1
                if not best_a or not best_j:
                    missed_ids[key] = datapoint
                scores[model]['best']['a'] += best_a
                scores[model]['best']['j'] += best_j
                scores[model]['best']['both'] += best_a and best_j
                if type(answers[model]['sensible']) == dict and len(answers[model]['sensible']) > 0:
                    # Sensible
                    intersection = set(answers[model]['sensible']['results']) & set(answers[model]['sensible']['correct'])
                    union = set(answers[model]['sensible']['results'] + answers[model]['sensible']['correct'])
                    scores[model]['sensible'] += len(intersection)/len(union)
                else:
                    scores[model]['sensible'] += 0
                modeltotals[model] += 1
        except Exception as e:
            print(f"Error processing model {model} for key {key}: {e}")
            malforms.append(key)
            continue
print(len(data))
desc_totals = {k: v for k, v in modeltotals.items() if 'desc' in k}
blind_totals = {k: v for k, v in modeltotals.items() if 'blind' in k}
other_totals = {k: v for k, v in modeltotals.items() if 'blind' not in k and 'desc' not in k}
desc_totals = dict(sorted(desc_totals.items()))
blind_totals = dict(sorted(blind_totals.items()))
other_totals = dict(sorted(other_totals.items()))
def print_modeltotals(mt):
    for model in mt:
        if mt[model] == len(data):
            print(f"\033[92m{model}: {mt[model]}/{len(data)}\033[0m")
        elif 'blind' in model and mt[model] >= int(len(data) * 0.15):
            print(f"\033[93m{model}: {mt[model]}/{len(data)}\033[0m")
        else:
            print(f"\033[91m{model}: {mt[model]}/{len(data)}\033[0m")
print_modeltotals(other_totals)
print("#"*50)
print_modeltotals(blind_totals)
print("#"*50)
print_modeltotals(desc_totals)

# Round to 2 dp, print with ampersand separated values, print modelname within {}

# Sort using ['best']['both'] as the key, descending
print("Blind Models:")
for model in sorted(scores):
    if model and 'blind' in model:
        total = modeltotals[model]
        #print(f"Model: {model}")
        print(f"{{{model.split()}}} & {round(scores[model]['best']['both'] / total * 100, 1)} & {round(scores[model]['best']['a'] / total * 100, 1)} & {round(scores[model]['best']['j'] / total * 100, 1)} & {round(scores[model]['sensible'] / total * 100, 1)}")
print("#"*50)
print("Desc Models:")
for model in sorted(scores):
    if model and 'desc' in model:
        total = modeltotals[model]
        #print(f"Model: {model}")
        print(f"{{{model.split()}}} & {round(scores[model]['best']['both'] / total * 100, 1)} & {round(scores[model]['best']['a'] / total * 100, 1)} & {round(scores[model]['best']['j'] / total * 100, 1)} & {round(scores[model]['sensible'] / total * 100, 1)}")
print("#"*50)
print("Other Models:")
for model in sorted(scores):
    if model and 'blind' not in model and 'desc' not in model:
        total = modeltotals[model]
        #print(f"Model: {model}")
        print(f"{{{model.split()}}} & {round(scores[model]['best']['both'] / total * 100, 1)} & {round(scores[model]['best']['a'] / total * 100, 1)} & {round(scores[model]['best']['j'] / total * 100, 1)} & {round(scores[model]['sensible'] / total * 100, 1)}")
        #print("-"*25)

print("Malformed IDs: ", malforms)
