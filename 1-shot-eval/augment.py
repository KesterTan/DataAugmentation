finalized_entries = []
for i in range(0, len(data) - 1, 2):
    example = data[i]
    instruction = data[i + 1]

    finalized_entry = {
        "question_id": i // 2 + 1,
        "text": (
            f"Example:\n\n"
            f"Instruction:\n{example['input']}\n\n"
            f"Use this API documentation for reference: {example['tools']}\n\n"
            f"Output:\n{example['output']}\n\n"
            f"Gold Answer:\n{example.get('gold_answer', 'N/A')}\n\n"
            f"Now solve:\n{instruction['input']}\n\n"
            f"Use this API documentation for reference: {instruction['tools']}"
        ),
        "category": "generic"
    }
    finalized_entries.append(finalized_entry)

# save to file
final_output_with_tools_path = "/mnt/data/nested_api_query_augmented_final.json"
with open(final_output_with_tools_path, "w") as f:
    json.dump(finalized_entries, f, indent=2)

final_output_with_tools_path

