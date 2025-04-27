from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

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

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

# --- Prepare model for k-bit training ---
model = prepare_model_for_kbit_training(model)

# --- Apply LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Load dataset ---
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- Tokenization function ---
def tokenize(example):
    prompt = example["code"]
    model_inputs = tokenizer(prompt, max_length=512, padding="max_length", truncation=True)
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
