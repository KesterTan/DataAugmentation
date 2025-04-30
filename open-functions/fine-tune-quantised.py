from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, AutoConfig
import torch
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from datasets import load_dataset
import bitsandbytes as bnb

# --- Settings ---
model_name = "gorilla-llm/gorilla-openfunctions-v2"
dataset_path = "../fine-tuning/first_1500_entries-first-fixed-serialized.jsonl"

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# --- Load model with 4-bit quantization ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Can use bfloat16 if you prefer
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.max_position_embeddings = 8912
config.rope_scaling = {"type": "linear", "factor": 2.0}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    config=config,
)

# --- Prepare model for k-bit training ---
model = prepare_model_for_kbit_training(model)

# --- Apply LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    max_position_embeddings=8912,
    rope_scaling={"type": "linear", "factor": 2.0}
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Load dataset ---
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- Tokenization function ---
def tokenize(example):
    prompt = example["code"]
    model_inputs = tokenizer(prompt, padding="max_length", truncation=True)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# --- Tokenize dataset ---
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names, batched=False)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    optim="adamw_torch_fused",
    report_to="none",  # Prevent wandb errors if not configured
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# --- Train ---
trainer.train()

# --- Save model and tokenizer ---
model.save_pretrained("fine-tuned-gorilla")
tokenizer.save_pretrained("fine-tuned-gorilla")