import json
import ast

INPUT_FILE = 'display-first-1860-train-rows.json'
OUTPUT_FILE = 'augmented-1860-custom-explanations.json'

def safe_load(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def format_args(args):
    return ", ".join([f"{k}={v}" for k, v in args.items()])

def build_reverse_chaining_explanation(calls, gold_answer):
    lines = []
    lines.append(f"We aim for the final answer **{gold_answer}**. Reverse Chaining starts from this goal and uncovers the prerequisite computations, step‑by‑step:")
    # reverse enumerate
    for idx, call in enumerate(reversed(calls), start=1):
        name = call.get('name')
        label = call.get('label', '')
        arg_str = format_args(call.get('arguments', {}))
        lines.append(f"{idx}. `{label}` is obtained via **{name}({arg_str})**.")
    lines.append("Running the chain forward with these intermediate values reproduces the gold answer.")
    return "\n".join(lines)

with open(INPUT_FILE, 'r') as f:
    raw_rows = json.load(f)

augmented_rows = []
for row in raw_rows:
    input_inst = row.get('input', '')
    tools_str = row.get('tools', '')
    calls_str = row.get('output', '')
    gold_answer = row.get('gold_answer', '')
    
    tools_json = safe_load(tools_str) or tools_str
    calls_json = safe_load(calls_str) or []
    
    explanation = build_reverse_chaining_explanation(calls_json, gold_answer)
    
    if calls_json:
        first_call = calls_json[0]
        first_desc = f"{first_call.get('name')}({format_args(first_call.get('arguments', {}))})"
    else:
        first_desc = "None"
    
    code_block = (
        f"###Instruction: {input_inst} Use this API documentation for reference: {tools_str}\n"
        f"###Output: <<<domain>>>: Multi‑step Numerical Reasoning\n"
        f"<<<api_call>>>: {first_desc}\n"
        f"<<<api_provider>>>: Internal Toolchain\n"
        f"<<<explanation>>>:\n{explanation}\n"
        f"<<<code>>>:\n{json.dumps(calls_json, indent=2)}\n"
        f"# Final gold answer\n{gold_answer}\n"
    )
    
    augmented_rows.append({
        "code": code_block,
        "api_call": first_desc,
        "provider": "Internal Toolchain",
        "api_data": {
            "tools": tools_json,
            "performance": {"dataset": "N/A", "accuracy": "N/A"}
        }
    })

with open(OUTPUT_FILE, 'w') as f:
    json.dump(augmented_rows, f, indent=2)

print("Saved to", OUTPUT_FILE)