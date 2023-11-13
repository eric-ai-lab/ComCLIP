import openai
import json
import os
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_arguments("openaikey", type=str)
parser.add_arguments("relation_json_path", type=str)
parser.add_arguments("data_path", type=str)
args = parser.parse_args()
data = pd.read_picklet(args.data_path)
SAVE_TEXT_JSON_PATH = args.relation_json_path+"/{}.json"


def get_relation(text):
    input = "Given a sentence in triple quotes \"\"\"{}\"\"\", analyze the objects in this sentence, the attributes of the objects ".format(text) \
        + "and how each objects connected. For example. for sentence \"\"\"young person sit on a boat\"\"\"" + \
        ", the output is {\"objects\": {\"person\" : {\"attributes\": \"young\"}, \"boat\":" + \
        "{\"attributes\": null}}, \"connections\": [{\"subject\": \"person\",\"verb\"" + \
        ": \"sit\", \"object\": \"boat\"}]}. Just output the json exactly like the format in example. "
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}], temperature=0)
    return completion["choices"][0]["message"]["content"]

if __name__ == "__main__":
    for idx, row in data.iterrows():
        if not os.path.exists(SAVE_TEXT_JSON_PATH.format(idx)):
            result = get_relation(row.sentence)
            with open(SAVE_TEXT_JSON_PATH.format(idx), 'w') as f:
                json.dump(result, f)

