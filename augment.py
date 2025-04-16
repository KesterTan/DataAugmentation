import json
import sys
import torch
import re
from transformers import LlamaTokenizer, LlamaForCausalLM


def load_llama_model(model_path):
    """
    Loads a local Llama model from the specified directory.
    """
    if model_path.endswith(".gguf"):
        raise ValueError("GGUF models are not supported with Hugging Face transformers. "
                         "Use `llama-cpp-python` or `ctransformers` instead.")
    
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
    return tokenizer, model


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
    Generates an augmented explanation for a given sample using a Llama model.
    """
    problem_statement = sample.get("input", "")
    chain_of_thought = sample.get("output", "")
    tools = sample.get("tools", "")
    gold_answer = sample.get("gold_answer", "")

    prompt = f"""You have the following problem statement:
    {problem_statement}
    {tools}

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
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Try to isolate the model output portion
    explanation_raw = decoded_output.split(prompt, 1)[-1].strip() if prompt in decoded_output else decoded_output.strip()

    return extract_json_from_output(explanation_raw)


def main(input_file, output_file, model_path):
    """
    Reads the input JSON, generates augmented explanations for each entry,
    and writes a new JSON file with the added data.
    """
    tokenizer, model = load_llama_model(model_path)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented_data = []
    for sample in data:
        explanation = generate_augmented_explanation(tokenizer, model, sample)
        sample["augmented_explanation"] = explanation
        augmented_data.append(sample)

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
