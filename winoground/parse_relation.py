from datasets import load_dataset
import openai
import json
import argparse
parser = argparse.ArgumentParser(description='Test BILP2 on winoground.')
parser.add_argument('--huggingface_token', type=str, help='Huggingface token from the Hugging Face account.')
parser.add_argument('--openai', type=str, help='OpenAI api key.')
parser.add_argument("--output_relation_path", type=str, help="Folder path that stores dense caption from GRiT")
args = parser.parse_args()
SAVE_TEXT_JSON_PATH = args.output_relation_path + "/{}_{}.json"
openai.api_key = args.openai

def load_preset(args):
    auth_token = args.huggingface_token
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    return winoground

def get_relation(text):
    input = "Given a sentence in triple quotes \"\"\"{}\"\"\", analyze the objects in this sentence, the attributes of the objects ".format(text) \
        + "and how each objects connected. For example, for sentence \"\"\"young person sit on a boat\"\"\"" + \
        ", the output is {\"objects\": {\"person\" : {\"attributes\": \"young\"}, \"boat\":" + \
        "{\"attributes\": null}}, \"connections\": [{\"subject\": \"person\",\"verb\"" + \
        ": \"sit\", \"object\": \"boat\"}]}. Just output the json exactly like the format in example. "
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}], temperature=0)
    return completion["choices"][0]["message"]["content"]

def parse_relation(winoground):
    for i in range(len(winoground)):
        if (i % 100) == 0:
            print("completed {}".format(i))
        result_0 = get_relation(winoground[i]["caption_0"])
        with open(SAVE_TEXT_JSON_PATH.format(i, 0), 'w') as f:
            json.dump(result_0, f)
        result_1 = get_relation(winoground[i]["caption_1"])
        with open(SAVE_TEXT_JSON_PATH.format(i, 1), 'w') as f:
            json.dump(result_1, f)

if __name__ == "__main__":
    winoground = load_preset(args)
    parse_relation(winoground)