import json

input_path = "display-all-training-data.json"
with open(input_path, "r") as f:
    data = json.load(f)

# Extract the last 360 entries (or first 1500)
rest = data[-361:]
# rest = data[:-360]

output_path = "split_original_last360.json"
with open(output_path, "w") as f:
    json.dump(rest, f, indent=2)

print(f"Extracted {len(rest)} entries to {output_path}")