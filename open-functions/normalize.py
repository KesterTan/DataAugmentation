import json

fixed_lines = []
with open("../fine-tuning/first_1500_entries-first.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        for tool in obj.get("api_data", {}).get("tools", []):
            output_params = tool.get("output_parameters", {}).get("properties", {})
            for key, param in output_params.items():
                if isinstance(param.get("type"), str):
                    param["type"] = [param["type"]]  # make it a list
        fixed_lines.append(json.dumps(obj))

with open("../fine-tuning/first_1500_entries-first-fixed.jsonl", "w") as f:
    for line in fixed_lines:
        f.write(line + "\n")
