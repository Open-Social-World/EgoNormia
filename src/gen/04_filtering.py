###########################

# PREREQUISITES:
# custom_data.json: File containing the final data
# custom_data_eval.json: File containing results of blind evaluation, this has to be done through the eval API

###########################
import json
import os


source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(f"{source_dir}/final_dataset/custom_data.json", "r") as f:
    data = json.loads(f.read())


blind_source = f"{source_dir}/final_dataset/custom_data_eval.json"
blind_target = f"{source_dir}/final_dataset/custom_data.json"

with open(blind_source, 'r') as f:
    json_data = json.load(f)


modelsource = 'blind_gemini-1.5-flash-002'

new_data = {}
new_data_eval = {}

for row in json_data:
    row_id = row
    row = json_data[row]
    if row == {}:
        new_data[row_id] = data[row_id]
        new_data_eval[row_id] = {}
        continue
    ans = row['answers'][modelsource]['best']['results']
    correct = row['answers'][modelsource]['best']['correct']

    if ans != correct:
        new_data[row_id] = data[row_id]
        new_data_eval[row_id] = {}

print(f"Filtered out {len(json_data) - len(new_data)} rows")
print(f"New data has {len(new_data)} rows")

# Save the filtered data to a new file
with open(blind_target, 'w') as f:
    json.dump(new_data, f, indent=4)

with open(blind_source, 'w') as f:
    json.dump(new_data_eval, f, indent=4)