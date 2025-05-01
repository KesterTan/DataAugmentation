import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────
# Model & quantization setup
# ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face model name
base_model_name = "gorilla-llm/gorilla-openfunctions-v2"

adapter_model_path = "./fine-tuned-gorilla-long-shifted"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    adapter_model_path,
    trust_remote_code=True
)

# Load base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load adapter weights (LoRA) on top of base model
model = PeftModel.from_pretrained(
    model,
    adapter_model_path,
    device_map="auto"
)
model.eval()

# ────────────────────────────────────────────────────────────────────
# I/O paths
# ────────────────────────────────────────────────────────────────────
input_json_path  = "../in-context-eval/last_360_entries.json"
output_json_path = "../in-context-eval_results-fine-tune-shifted.json"

with open(input_json_path, "r") as f:
    data = json.load(f)

# ────────────────────────────────────────────────────────────────────
# Inference loop
# ────────────────────────────────────────────────────────────────────
outputs = []

for sample in tqdm(data, desc="Running inference"):
    if "text" in sample:
        prompt = sample["text"]
    elif "instruction" in sample:
        prompt = sample["instruction"]
    else:
        raise ValueError("Sample missing 'text' or 'instruction' field")

    system = "You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."

    prompt_text = (
        "{system}\n### Instruction: <<question>> " + prompt.strip() +
        "\n### Response:"
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[1]
    answer_ids = generated_ids[0][prompt_length:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    outputs.append({"input": prompt, "output": answer_text})
# ────────────────────────────────────────────────────────────────────
# Save results
# ────────────────────────────────────────────────────────────────────
with open(output_json_path, "w") as f:
    json.dump(outputs, f, indent=2)

print(f"Saved {len(outputs)} inference outputs to {output_json_path}")
