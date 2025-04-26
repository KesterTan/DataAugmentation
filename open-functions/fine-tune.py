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

dataset = load_dataset("json", data_files="../fine-tuning/first_1500_entries_normalized.jsonl", split="train")

def tokenize(example):
    prompt = example["input"]
    inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    return {key: val.squeeze() for key, val in inputs.items()}

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./fine-tuned-gorilla",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none"
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
