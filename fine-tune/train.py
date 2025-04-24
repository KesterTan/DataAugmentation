from transformers import AutoConfig, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

config = AutoConfig.from_pretrained("../../gorilla-7b-hf-v0/config.json")
model_directory = "../../gorilla-7b-hf-v0/"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 
