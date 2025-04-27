import json
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
binarizer = MultiLabelBinarizer()

result = "in-context-eval_results.json"

# result = "1-shot-eval/try_two_results.json"

df = pd.read_parquet("hf://datasets/ibm-research/nestful/data/train-00000-of-00001.parquet")

nestful_data = df[-361:]

with open(result, "r") as f:
    result_data = json.load(f)


data = []

def parse_func(text, label):
    p = text.find('(')
    if p == -1:
        print("Parsing error")
        print("Parsing error", "text")
    
    function = {"name" : text[:p]}
    output = []

    params = text[p+1:len(text) - 1].split(", ")
    args = {}

    i = 0
    while i < len(params):
        params[i] = params[i].strip()
        e = params[i].find("=")
        if e == -1:
            print("Parsing error", "text")

        key = params[i][:e].strip()
        value = params[i][e+1:].strip()

        if "(" in value:
            i += 1
            count = 1
            nested = value
            while count != 0:
                if ')' in params[i]:
                    count -= 1
                elif '(' in params[i]:
                    count += 1
                nested += ", " + params[i]
                i += 1
            
            res = parse_func(nested, label)
            output = res

            value = "var" + str(label) + ".output"
            label += 1
        else:
            i += 1
        
        args[key] = value
    

    function["arguments"] = args
    function["label"] = "var" + str(label)

    return output + [function]


for i in range(len(result_data)):

    input = result_data[i]["input"]
    pred_output = result_data[i]["output"]

    index = input.find('Use this API documentation')
    input = input.strip()[:index-4]
    # if not (nestful_data['input'].iloc[i] == input):
    #     continue

    pred_output = pred_output.strip()[12:]

    gold_output = json.loads(nestful_data['output'].iloc[i])
    gold_func_calls = [gold_output[i]["name"] for i in range(len(gold_output))]

    if "Error" in pred_output:
        continue

    pred_output = pred_output.split("<<function>>")
    
    pred_dict_list = []
    for pred in pred_output:
        print(pred)
        try: 
            pred_dict_list += parse_func(pred, len(pred_dict_list))
        except:        
            print("oopsies")

    data.append(pred_dict_list)

output_path = "metrics/parsed.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

output_path

    # f1_score(binarizer.transform(real_output),
    #                                   binarizer.transform(pred_output),
    #                                   average='macro')

    





    # print(response)
    # index = response.find("import")
    # print(index)
    # print("\n")
    # .split('\\n')
    # if not index == -1:



# F1 scores for function names

# F1 scores for parameter name

# Partial Sequence Matching Accuracy

# Full Sequence Matching Accuracy

# Win Rate
