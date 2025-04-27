from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoConfig
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "gorilla-llm/gorilla-openfunctions-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.max_position_embeddings = 4096

model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, device_map="auto")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                     # Increase rank
    lora_alpha=32,             # Increase alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    use_longlora=True,         # 👈 This activates LongLoRA
    max_position_embeddings=4096 # 👈 Extend context length
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="../fine-tuning/first_1500_entries-first-fixed-serialized.jsonl", split="train")

def tokenize(example):
    prompt = example["code"]
    inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=4096, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["labels"][inputs["attention_mask"] == 0] = -100
    return {key: val.squeeze() for key, val in inputs.items()}

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,     # Try 2 if memory allows
    gradient_accumulation_steps=8,     # Keep effective batch size
    learning_rate=5e-5,                # Lower LR for more stability
    num_train_epochs=5,                # 5 epochs (you don't have many examples)
    warmup_ratio=0.1,                  # Warmup helps convergence
    lr_scheduler_type="cosine",         # Cosine scheduler helps
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    bf16=False,
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("fine-tuned-gorilla-v2")
tokenizer.save_pretrained("fine-tuned-gorilla-v2")
