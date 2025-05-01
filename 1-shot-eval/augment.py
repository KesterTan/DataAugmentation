import json

# Load data from JSON file
input_path = ""  # Change this path as needed
with open(input_path, "r") as f:
    data = json.load(f)

# Reformat and append tools after the "Now solve" prompt
finalized_entries = []

for i in range(0, len(data) - 1, 2):
    example = data[i]
    instruction = data[i + 1]

    finalized_entry = {
        "question_id": (i // 2) + 1,
        "text": (
            f"Example:\n\n"
            f"Instruction:\n{example.get('input', '')}\n\n"
            f"Use this API documentation for reference: {example.get('tools', 'N/A')}\n\n"
            f"Output:\n{example.get('output', '')}\n\n"
            f"Gold Answer:\n{example.get('gold_answer', 'N/A')}\n\n"
            f"Now solve:\n{instruction.get('input', '')}\n\n"
            f"Use this API documentation for reference: {instruction.get('tools', 'N/A')}"
        ),
        "category": "generic"
    }

    finalized_entries.append(finalized_entry)

# Save to file
final_output_with_tools_path = "/mnt/data/nested_api_query_augmented_final.json"
with open(final_output_with_tools_path, "w") as f:
    json.dump(finalized_entries, f, indent=2)

print(f"Saved to {final_output_with_tools_path}")
