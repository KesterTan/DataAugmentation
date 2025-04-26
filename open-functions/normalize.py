import json

input_file = "DataAugmentation/fine-tuning/first_1500_entries_fixed.jsonl"
output_file = "DataAugmentation/fine-tuning/first_1500_entries_normalized.jsonl"

def normalize_type_field(entry):
    try:
        tools = entry.get("api_data", {}).get("tools", [])
        for tool in tools:
            output_params = tool.get("output_parameters", {})
            properties = output_params.get("properties", {})
            output_0 = properties.get("output_0", {})
            output_type = output_0.get("type")

            # If type is a string, make it a list
            if isinstance(output_type, str):
                output_0["type"] = [output_type]

            # If type missing, do nothing
    except Exception as e:
        print(f"Error normalizing entry: {e}")
    return entry

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        entry = json.loads(line)
        entry = normalize_type_field(entry)
        outfile.write(json.dumps(entry) + "\n")

print("✅ Fixed and saved to", output_file)
