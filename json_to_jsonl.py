import json

file = "1-shot-eval/last_360_entries"
# Load the JSON array
with open(file+".json", "r") as f:
    data = json.load(f)

# Write it as JSONL
with open(file+".jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
