import json
import pandas as pd
from tqdm import tqdm

nestful = "testing/nestful_data.json"
result = "testing/results_last_360.json"

df = pd.read_parquet("hf://datasets/ibm-research/nestful/data/train-00000-of-00001.parquet")

nestful_data = df[-361:]

print(df[-361:]['output'].iloc[0])

s = df[-361:]['output'].iloc[0]

json_array = json.loads(s)

for d in json_array:
    print(d)




# with open(nestful, "r") as f:
#     nestful_data = json.load(f)

with open(result, "r") as f:
    result_data = json.load(f)



# for d in nestful_data[-360:][:10]:

#     # sample_id, input, output, tools, gold_answer
#     # print(d["input"])
#     # print(d["input"])
#     output = d['output']
#     # functions = [output[i]["name"] for i in range(len(output))]
#     # print(functions)
#     # for function in output:

for i in range(5):

    question = result_data[i]["questions"]
    response = result_data[i]["response"]

    index = question.find('Use this API documentation')
    input = question.strip()[9:index - 4]
    if not (nestful_data['input'].iloc[i] == input):
        print("ummmm")
        continue


    real_output = json.loads(nestful_data['output'].iloc[i])
    functions = [real_output[i]["name"] for i in range(len(real_output))]
    pred_output = "" # FILL THIS IN LATER????
    correct_count = 0
    for f in functions:
        if f in pred_output:
            correct_count += 0

    



    index = response.find("<<<code>>>")


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
