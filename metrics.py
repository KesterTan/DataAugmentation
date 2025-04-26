import json
from tqdm import tqdm

nestful = "testing/nestful_data.json"
result = "testing/results_last_360.json"

with open(nestful, "r") as f:
    nestful_data = json.load(f)

with open(result, "r") as f:
    result_data = json.load(f)

for d in nestful_data[-360:][:10]:

    # sample_id, input, output, tools, gold_answer
    print(d["input"])
    # print(d["input"])
    output = d["output"]
    functions = [output[i]["name"] for i in range(len(output))]
    # print(functions)
    # for function in output:

print("\n")

for d in result_data[:10]:
    question = d["questions"]
    index = question.find('Use this API documentation')
    input = question.strip()[9:index - 4]

    response = d["response"]

    # print(response)
    # index = response.find("import")
    print(input)
    # print(index)
    # print("\n")
    # .split('\\n')
    # if not index == -1:



# F1 scores for function names

# F1 scores for parameter name

# Partial Sequence Matching Accuracy

# Full Sequence Matching Accuracy

# Win Rate
