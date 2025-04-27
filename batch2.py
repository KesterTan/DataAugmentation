import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig          # NEW
)
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────
# Model & quantization setup
# ────────────────────────────────────────────────────────────────────
# Load model and tokenizer with 4-bit quantization using bitsandbytes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gorilla-llm/gorilla-openfunctions-v2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 4-bit weights
    bnb_4bit_quant_type="nf4",          # "nf4" or "fp4"
    bnb_4bit_use_double_quant=True,     # second quantization step on weights
    bnb_4bit_compute_dtype=torch.bfloat16  # computation in bf16 (fp16 also works)
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",                  # let Accelerate split layers across GPUs / RAM
    trust_remote_code=True
)
model.eval()

# ────────────────────────────────────────────────────────────────────
# I/O paths
# ────────────────────────────────────────────────────────────────────
input_json_path  = "in-context-eval/try_two.json"   # your source file
output_json_path = "inference_outputs.json"         # where to write results

with open(input_json_path, "r") as f:
    data = json.load(f)

# ────────────────────────────────────────────────────────────────────
# Inference loop
# ────────────────────────────────────────────────────────────────────
outputs = []

for sample in tqdm(data, desc="Running inference"):
    # Accept either "text" or "instruction" keys
    if "text" in sample:
        prompt = sample["text"]
    elif "instruction" in sample:
        prompt = sample["instruction"]
    else:
        raise ValueError("Sample missing 'text' or 'instruction' field")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,     # adjust to your needs
            do_sample=False         # greedy / deterministic decoding
        )

    outputs.append({
        "input":  prompt,
        "output": tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    })

# ────────────────────────────────────────────────────────────────────
# Save results
# ────────────────────────────────────────────────────────────────────
with open(output_json_path, "w") as f:
    json.dump(outputs, f, indent=2)

print(f"Saved {len(outputs)} inference outputs to {output_json_path}")
