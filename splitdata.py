import json

# Path to the original JSON file
input_path = "display-all-training-data.json"

# Load the JSON data from the file
with open(input_path, "r") as f:
    data = json.load(f)

# Extract the last 360 entries
rest = data[-360:]
# rest = data[:-360]

# Path to the output JSON file
output_path = "updated_last_360.json"

# Save the last 360 entries to a new file
with open(output_path, "w") as f:
    json.dump(rest, f, indent=2)

print(f"Extracted {len(rest)} entries to {output_path}")
