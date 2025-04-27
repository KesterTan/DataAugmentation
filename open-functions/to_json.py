# Redo the fix — but this time fix BOTH `parameters` and `output_parameter`
fixed_entries = []
broken_count = 0

def normalize_type_field(param_obj):
    """Recursively fix the type fields."""
    global broken_count
    if isinstance(param_obj, dict):
        for key, val in param_obj.items():
            if isinstance(val, dict):
                normalize_type_field(val)
            elif key == 'type':
                if isinstance(val, list):
                    # Convert list to string, e.g., ["int", "float"] -> "int or float"
                    param_obj[key] = ' or '.join(val)
                    broken_count += 1

# Read, fix, and re-save
with open(file_path, 'r') as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            tools = obj.get('api_data', {}).get('tools', [])
            for tool in tools:
                # Fix input parameters
                params = tool.get('parameters', {})
                normalize_type_field(params)
                # Fix output parameters
                output_params = tool.get('output_parameter', {})
                normalize_type_field(output_params)
            fixed_entries.append(obj)

# Save the corrected file
fixed_file_path_v2 = '../fine-tuning/first_1500_entries-first-fixed-v2.jsonl'
with open(fixed_file_path_v2, 'w') as f:
    for entry in fixed_entries:
        f.write(json.dumps(entry) + '\n')

broken_count, fixed_file_path_v2
