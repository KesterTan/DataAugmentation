import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Set the model name
model_name = "gorilla-llm/gorilla-openfunctions-v2"

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# Load the dataset
input_json_path = "in-context-eval/try_two.json"  # path to your input JSON
output_json_path = "inference_outputs.json"  # path to save outputs

with open(input_json_path, "r") as f:
    data = json.load(f)

# Inference loop
outputs = []

for sample in tqdm(data, desc="Running Inference"):
    # Change this depending on your JSON structure
    # Assuming each sample has a 'text' or 'instruction' field
    if "text" in sample:
        input_text = sample["text"]
    elif "instruction" in sample:
        input_text = sample["instruction"]
    else:
        raise ValueError("Sample missing 'text' or 'instruction' field")

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,  # you can adjust depending on expected output size
            do_sample=False,  # deterministic output
        )
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Save the original + the output
    outputs.append({
        "input": input_text,
        "output": output_text
    })

# Save all outputs
with open(output_json_path, "w") as f:
    json.dump(outputs, f, indent=2)

print(f"Saved {len(outputs)} inference outputs to {output_json_path}")
