from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

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
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="../fine-tuning/first_1500_entries-first-fixed-serialized.jsonl", split="train")

def tokenize(example):
    prompt = example["code"]
    inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    return {key: val.squeeze() for key, val in inputs.items()}

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

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
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("fine-tuned-gorilla")
tokenizer.save_pretrained("fine-tuned-gorilla")
