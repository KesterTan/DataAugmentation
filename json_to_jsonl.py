import json

file = "1-shot-eval/last_360_entries"

with open(file+".json", "r") as f:
    data = json.load(f)

with open(file+".jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
