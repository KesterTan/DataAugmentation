from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, AutoConfig
import torch
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from datasets import load_dataset
import bitsandbytes as bnb
import torch.nn as nn
import math

# --- Settings ---
model_name = "gorilla-llm/gorilla-openfunctions-v2"
dataset_path = "../fine-tuning/first_1500_entries-first-fixed-serialized.jsonl"

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# --- BitsAndBytes config for 4-bit quantization ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# --- RoPE Scaling + Long Context Support ---
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.max_position_embeddings = 8912
config.rope_scaling = {"type": "linear", "factor": 2.0}

# --- Load quantized model ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    config=config,
)

# --- Inject Sparse Attention ---
def create_sliding_window_mask(seq_len, window_size):
    mask = torch.full((seq_len, seq_len), float("-inf"))
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 0
    return mask

class SparseAttention(nn.Module):
    def __init__(self, original_attn, window_size=128):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window_size = window_size

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, seq_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        sparse_mask = create_sliding_window_mask(seq_len, self.window_size).to(attn_scores.device)
        attn_scores = attn_scores + sparse_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

def patch_model_with_sparse_attention(model):
    for name, module in model.model.named_modules():
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            parent_module = dict(model.model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model.model
            setattr(parent_module, name.split('.')[-1], SparseAttention(module))

patch_model_with_sparse_attention(model)

# --- Prepare for k-bit training + apply LoRA ---
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Load dataset ---
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- Tokenize ---
def tokenize(example):
    prompt = example["code"]
    model_inputs = tokenizer(prompt, padding="max_length", truncation=True)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names, batched=False)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="./results-long",
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
    report_to="none",
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

# --- Save model ---
model.save_pretrained("fine-tuned-gorilla-long")
tokenizer.save_pretrained("fine-tuned-gorilla-long")
