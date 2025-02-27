import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noblind', type=str, required=False, help='Report blind')
parser.add_argument('--file', type=str, required=True, help='Target eval file')
args = parser.parse_args()

if args.noblind:
    noblind = True
else:
    noblind = False

file = args.file

with open(f'src/final_dataset/{file}.json', 'r') as f:
    data = json.load(f)

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
                'best': {'a': 0, 'j': 0, 'both': 0},
                'sensible': 0
            }
        modeltotals[model] = 0
        
what = {}

for key in data:
    datapoint = data[key]

    answers = datapoint
    for model in sorted(list(answers.keys())):
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

for model in sorted(scores):
    if model and 'blind' in model:
        total = modeltotals[model]
        print(f"Model: {model}")
        print(f"Best: {scores[model]['best']['a'] / total * 100}, {scores[model]['best']['j'] / total * 100}, {scores[model]['best']['both'] / total * 100}")
        print(f"Sensible: {scores[model]['sensible'] / total * 100}")

print("#"*50)
for model in sorted(scores):
    if model and 'desc' in model:
        total = modeltotals[model]
        print(f"Model: {model}")
        print(f"Best: {scores[model]['best']['a'] / total * 100}, {scores[model]['best']['j'] / total * 100}, {scores[model]['best']['both'] / total * 100}")
        print(f"Sensible: {scores[model]['sensible'] / total * 100}")
print("#"*50)
for model in sorted(scores):
    if model and 'blind' not in model and 'desc' not in model:
        total = modeltotals[model]
        print(f"Model: {model}")
        print(f"Best: {scores[model]['best']['a'] / total * 100}, {scores[model]['best']['j'] / total * 100}, {scores[model]['best']['both'] / total * 100}")
        print(f"Sensible: {scores[model]['sensible'] / total * 100}")