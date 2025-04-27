# batch_inference_last360.py
"""
Run batched, fp16 inference with Gorilla-OpenFunctions-v2 on every row of
last_360_entries.jsonl, adding a 'response' field and writing one JSON line
per record to last_360_entries_responses.jsonl.
"""

import json, torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. CONFIG ---------------------------------------------------------------
MODEL_NAME  = "gorilla-llm/gorilla-openfunctions-v2"
DATA_PATH   = "in-context-eval/try_two.jsonl"      # <-- update if you move it
OUT_PATH    = "in-context-eval/last_360_entries_responses.jsonl"
PROMPT_COL  = "text"      # column that holds the natural-language prompt
BATCH_SIZE  = 8           # adjust to fit your GPU
MAX_NEW     = 256         # max tokens to *generate* (does not trim input)
# -----------------------------------------------------------------------------


# 2. Load tokenizer & model (fp16 on all available GPUs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = (AutoModelForCausalLM
         .from_pretrained(MODEL_NAME,
                          torch_dtype=torch.float16,
                          device_map="auto")
         .eval())          # inference mode


# 3. Load dataset as a stream (handles very large files with low RAM use)
ds = load_dataset("json", data_files=DATA_PATH, split="train", streaming=True)


# 4. Generate in mini-batches and write results
with open(OUT_PATH, "w") as fout:
    batch_prompts, batch_meta = [], []   # meta = everything except the prompt

    def flush_batch():
        """Tokenize, generate, append answers, write to disk, then clear batch."""
        enc = tokenizer(batch_prompts, padding=True, truncation=True,
                        return_tensors="pt").to(model.device)

        with torch.no_grad():
            outs = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=False,                      # greedy decoding
                eos_token_id=tokenizer.eos_token_id
            )

        prompt_len = enc["input_ids"].shape[1]        # strip prompt tokens
        for meta, out in zip(batch_meta, outs):
            meta["response"] = tokenizer.decode(
                out[prompt_len:], skip_special_tokens=True
            )
            fout.write(json.dumps(meta) + "\n")

    # Iterate through the file
    for record in tqdm(ds, desc="Generating"):
        batch_prompts.append(record[PROMPT_COL])
        # keep every other key (question_id, category, …)
        batch_meta.append({k: v for k, v in record.items() if k != PROMPT_COL})

        if len(batch_prompts) == BATCH_SIZE:
            flush_batch()
            batch_prompts, batch_meta = [], []

    # final partial batch
    if batch_prompts:
        flush_batch()

print(f"✓ Finished. Wrote responses to {OUT_PATH}")

