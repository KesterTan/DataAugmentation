import json
import sys
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def load_llama_model(model_path):
    """
    Loads a local Llama-7B model from the specified directory.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
    return tokenizer, model

def generate_augmented_explanation(tokenizer, model, sample):
    """
    Generates an augmented explanation for a given sample using a Llama-7B model.
    """
    sample_id = sample.get("sample_id")
    problem_statement = sample.get("input")
    chain_of_thought = sample.get("output")  
    tools = sample.get("tools")
    gold_answer = sample.get("gold_answer")

    prompt = f"""You have the following problem statement:
        {problem_statement}

        Chain of Thought (in JSON):
        {chain_of_thought}

        Tools (in JSON):
        {tools}

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

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    # Generate the augmented explanation with the model
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Attempt to isolate only the JSON portion that comes after our prompt.
    if prompt in decoded_output:
        explanation_raw = decoded_output.split(prompt, 1)[-1].strip()
    else:
        explanation_raw = decoded_output.strip()

    # Attempt to parse the explanation as JSON. 
    # If it fails, fall back to storing the raw string.
    try:
        explanation_json = json.loads(explanation_raw)
    except json.JSONDecodeError:
        explanation_json = {
            "error": "Invalid JSON from model",
            "raw_output": explanation_raw
        }

    return explanation_json


def main(input_file, output_file, model_path):
    """
    Reads the input JSON, generates augmented explanations for each entry,
    and writes a new JSON file with the added data.
    """
    # 1. Load the model & tokenizer
    tokenizer, model = load_llama_model(model_path)

    # 2. Read input JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Process each sample
    augmented_data = []
    for sample in data:
        explanation = generate_augmented_explanation(tokenizer, model, sample)

        # Attach the augmented explanation to the sample under a new key
        sample["augmented_explanation"] = explanation
        augmented_data.append(sample)

    # 4. Write to output JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2)

    print(f"Augmented data has been saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python augment_data.py <input_json> <output_json> <model_path>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]
    llama_model_path = sys.argv[3]

    main(input_json, output_json, llama_model_path)
