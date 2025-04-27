import json
import os

input_file = "in-context-eval/try_two.jsonl"
output_file = "in-context-eval/try_two.json"

# Ensure the output folder exists
os.makedirs(output_file, exist_ok=True)

# Determine the output JSON filename
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_json_file = os.path.join(output_file, base_name + '.json')

# Read the JSONL file and aggregate the data
data = []
with open(input_file, 'r') as jsonl_file:
    for line_number, line in enumerate(jsonl_file, start=1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            continue

# Write to the JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Converted {input_file} to {output_json_file}")
