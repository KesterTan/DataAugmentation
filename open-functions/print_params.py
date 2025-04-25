from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model_name = "gorilla-llm/gorilla-openfunctions-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Common for Gorilla (LLama-like)
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
