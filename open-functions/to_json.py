import json

input_file = "DataAugmentation/fine-tuning/first_1500_entries.json"
output_file = "DataAugmentation/fine-tuning/first_1500_entries_fixed.jsonl"

# Read the JSON array
with open(input_file, "r") as f:
    data = json.load(f)

# Write each entry as a separate line
with open(output_file, "w") as f:
    for entry in data:
        f.write(json.dumps(entry) + "\n")
