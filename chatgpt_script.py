"""
compute_nestful_metrics.py
Run:  python compute_nestful_metrics.py split_original_last360.json in-context-eval_results.json
"""
import json, sys, re
from collections import Counter

# ---------- helpers ----------
def extract_calls_from_gold(raw):
    # gold is a JSON string representing a list[dict]
    return json.loads(raw)

def extract_calls_from_pred(raw):
    """
    Parses predictions for function calls of the form:
    name(arg=val, ...)
    """
    fn_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    param_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=')

    calls, current = [], None
    depth = 0
    token = ''
    for ch in raw:
        if ch == '(':
            name = token.strip()
            current = {'name': name, 'arguments': {}}
            token, depth = '', 1
        elif ch == ',' and depth == 1:
            if '=' in token:
                key, val = token.split('=', 1)
                current['arguments'][key.strip()] = val.strip()
            token = ''
        elif ch == ')' and depth == 1:
            if token.strip():
                if '=' in token:
                    key, val = token.split('=', 1)
                    current['arguments'][key.strip()] = val.strip()
            if current:
                calls.append(current)
            token, depth, current = '', 0, None
        else:
            token += ch
    return calls

def f1(pred_items, gold_items):
    p, g = Counter(pred_items), Counter(gold_items)
    tp = sum((p & g).values())
    precision = tp / sum(p.values()) if p else 0
    recall    = tp / sum(g.values()) if g else 0
    return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

# ---------- main ----------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <gold_path> <pred_path>")
        sys.exit(1)

    gold_path, pred_path = sys.argv[1], sys.argv[2]
    with open(gold_path) as f:
        gold = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)

    assert len(gold) == len(pred), "Mismatch in number of examples between gold and pred."

    fn_F1s, param_F1s = [], []
    partial_correct, full_correct, win_correct = 0, 0, 0

    for g, p in zip(gold, pred):
        gold_calls = extract_calls_from_gold(g['output'])
        pred_calls = extract_calls_from_pred(p['output'])

        gold_fns = [c['name'] for c in gold_calls]
        pred_fns = [c['name'] for c in pred_calls]
        fn_F1s.append(f1(pred_fns, gold_fns))

        gold_params = [k for c in gold_calls for k in c['arguments'].keys()]
        pred_params = [k for c in pred_calls for k in c['arguments'].keys()]
        param_F1s.append(f1(pred_params, gold_params))

        # partial sequence
        gold_serialized = {json.dumps(c, sort_keys=True) for c in gold_calls}
        pred_serialized = {json.dumps(c, sort_keys=True) for c in pred_calls}
        partial_correct += int(pred_serialized.issubset(gold_serialized))

        # full sequence (order-agnostic comparison of serialized sets)
        full_correct += int(pred_serialized == gold_serialized)

        # win-rate
        if pred_serialized and pred_serialized.issubset(gold_serialized) and (pred_serialized == gold_serialized):
            win_correct += int(p.get('gold_answer') == g.get('gold_answer'))

    N = len(gold)
    print("\n--- NESTful metrics ---")
    print(f"F1 (Function names):       {sum(fn_F1s) / N:.4f}")
    print(f"F1 (Parameter names):      {sum(param_F1s) / N:.4f}")
    print(f"Partial-sequence accuracy: {partial_correct / N:.4f}")
    print(f"Full-sequence accuracy:    {full_correct / N:.4f}")
    print(f"Win-rate:                  {win_correct / N:.4f}")
