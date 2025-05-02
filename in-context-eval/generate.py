import json
import pandas as pd
from pathlib import Path

# Load the JSON file
file_path = Path("display-first-1860-train-rows.json")
with open(file_path, "r") as file:
    data = json.load(file)

# Transform each record into the required output format
formatted_data = []
for idx, row in enumerate(data):
    formatted_data.append({
        "question_id": idx + 1,
        "text": f"{row['input']}\\\n \nUse this API documentation for reference: {row['tools']}\\\n \n",
        "category": "generic"
    })

# Save to a JSON file
output_path = "augmented_nested_api_questions.json"
with open(output_path, "w") as f:
    json.dump(formatted_data, f, indent=2)

output_path

