import json
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_json_from_output(output):
    """
    Attempts to extract the first JSON-like object from the model output.
    """
    try:
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass

    return {
        "error": "Invalid JSON from model",
        "raw_output": output
    }

def generate_augmented_explanation(tokenizer, model, sample):
    """
    Generates an augmented explanation using the locally loaded DeepSeek model.
    """
    problem_statement = sample.get("input", "")
    chain_of_thought = sample.get("output", "")
    tools = sample.get("tools", "")
    gold_answer = sample.get("gold_answer", "")

    prompt = f"""You have the following problem statement:
    {problem_statement} 
    Tools: {tools}

    Chain of Thought (in JSON):
    {chain_of_thought}

    Gold Answer:
    {gold_answer}

    Please produce an **augmented explanation** in the following JSON style:

    ###Instruction: Provide a short user-level instruction about the problem and how it is solved.
    ###Output: 
    <<<domain>>>: The domain or context (e.g. 'Text language model'),
    <<<api_call>>>: Example or relevant function used,
    <<<api_provider>>>: The library or environment used,
    <<<explanation>>>: Provide a clarifying explanation,
    <<<code>>>: Provide a relevant code snippet, wrapped in triple backticks.

    The final response **must** be valid JSON.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Attempt to remove the original prompt from the output
    if prompt in decoded_output:
        decoded_output = decoded_output.split(prompt, 1)[-1].strip()

    return extract_json_from_output(decoded_output)

def main(input_file, output_file):
    """
    Reads the input JSON, generates augmented explanations for each entry,
    and writes a new JSON file with the added data.
    """
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    # Read the input JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented_data = []
    for sample in data:
        explanation = generate_augmented_explanation(tokenizer, model, sample)
        sample["augmented_explanation"] = explanation
        augmented_data.append(sample)

    # Write the output JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2)

    print(f"Augmented data has been saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python augment_data.py <input_json> <output_json>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]

    main(input_json, output_json)
