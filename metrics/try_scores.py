import json
from collections import Counter
from typing import List, Dict

# --- Load Data ---
with open('../one-shot-eval_results.json') as f:
    ground_truth = json.load(f)

with open('./results2_last_360.json') as f:
    predictions = json.load(f)

# --- Extractors ---
def extract_functions(output_str):
    try:
        data = json.loads(output_str)
        if isinstance(data, list):
            return data
        else:
            return []
    except:
        return []


def extract_parameters(output_str):
    try:
        params = []
        for call in json.loads(output_str):
            if 'arguments' in call:
                params.extend(call['arguments'].keys())
        return params
    except:
        return []

def extract_full_sequence(output_str):
    try:
        return json.loads(output_str)
    except:
        return []

# --- F1 Calculation ---
def f1(pred, gold):
    pred_counter = Counter(pred)
    gold_counter = Counter(gold)
    tp = sum((pred_counter & gold_counter).values())
    precision = tp / len(pred) if pred else 0
    recall = tp / len(gold) if gold else 0
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

# --- Metrics Storage ---
function_f1s = []
param_f1s = []
partial_matches = []
full_matches = []
win_rates = []

# --- Matching ---
for pred_item, gold_item in zip(predictions, ground_truth):
    pred_output = pred_item.get('response', '')
    gold_output = gold_item.get('output', '')

    # Function names
    pred_funcs = extract_functions(pred_output)
    gold_funcs = extract_functions(gold_output)
    function_f1s.append(f1(pred_funcs, gold_funcs))

    # Parameter names
    pred_params = extract_parameters(pred_output)
    gold_params = extract_parameters(gold_output)
    param_f1s.append(f1(pred_params, gold_params))

    # Partial sequence match
    pred_seq = extract_full_sequence(pred_output)
    gold_seq = extract_full_sequence(gold_output)
    
    if not isinstance(pred_seq, list):
        pred_seq = []
    if not isinstance(gold_seq, list):
        gold_seq = []
    
    gold_calls_set = {(call['name'], frozenset(call.get('arguments', {}).items())) for call in gold_seq}
    pred_calls_set = {(call['name'], frozenset(call.get('arguments', {}).items())) for call in pred_seq}
    
    partial_match = any(call in gold_calls_set for call in pred_calls_set)
    partial_matches.append(int(partial_match))

    # Full sequence match
    full_match = (pred_calls_set == gold_calls_set)
    full_matches.append(int(full_match))

    # Win rate
    all_pred_valid = all('name' in call and 'arguments' in call for call in pred_seq)
    correct_execution = full_match  # Simulate execution matching
    win = int(all_pred_valid and correct_execution)
    win_rates.append(win)

# --- Aggregate Results ---
print(f"F1 Score for Function Names: {sum(function_f1s)/len(function_f1s):.4f}")
print(f"F1 Score for Parameter Names: {sum(param_f1s)/len(param_f1s):.4f}")
print(f"Partial Sequence Matching Accuracy: {sum(partial_matches)/len(partial_matches):.4f}")
print(f"Full Sequence Matching Accuracy: {sum(full_matches)/len(full_matches):.4f}")
print(f"Win Rate: {sum(win_rates)/len(win_rates):.4f}")
