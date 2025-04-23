from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gorilla-llm/gorilla-7b-hf-v1-gguf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                   
    lora_alpha=32,           
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,      
    bias="none",              
    task_type="CAUSAL_LM"
)

# Wrap the model
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
