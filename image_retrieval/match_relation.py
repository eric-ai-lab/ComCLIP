import openai
import json
import pandas as pd
import os
from json import JSONDecodeError

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--dense_caption_path", type=str)
parser.add_argument("--text_relation_path", type=str)
parser.add_argument("--openai", type=str, help="openai_key")
args = parser.parse_args()

DATA_PICKLE_PATH = args.dataset 
data = pd.read_pickle(DATA_PICKLE_PATH)
DENSE_CAPTION_FILE_PATH = args.dense_caption_path + "/{}.json"
TEXT_RELATION_FILE_PATH = args.text_relation_path + "/{}.json"
SAVE_MATCHING_PATH = 'matched_relation/row_{}_image_{}.json'

openai.api_key = args.openai

def get_dense_caption(image_id):
    f = open(DENSE_CAPTION_FILE_PATH.format(image_id))
    result = list(json.load(f).keys())
    return result

def get_sentence_json(row_id):
    f = open(TEXT_RELATION_FILE_PATH.format(row_id))
    try:
        result = json.load(f)
        if type(result) == dict:
            return list(result["objects"].keys())
        else:
            return list(json.loads(result)["objects"].keys())
    except Exception as e:
        return list(json.loads(result)["objects"].keys())

def match_objects(sentence, object, list):
    try:
        input = "Given a list of labels in triple quotes \"\"\"{}\"\"\" ".format(list) + \
                "and a sentence in triple quotes \"\"\"{}\"\"\". Tell me which labels in the list could refer to ".format(sentence) + \
                "the \"{}\" from the sentence. Only output the results in json format with 1 key named labels. If none of the labels refer to the \"{}\", set the value in the json to be an empty list".format(object, object)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}], temperature=0)
        return json.loads(completion["choices"][0]["message"]["content"])
    except Exception as e:
        raise e
    
def generate_caption_to_image(caption, obejct_list, image_list):
    result = {}
    for object in obejct_list:
        matched = match_objects(caption, object, image_list)["labels"]
        result[object] = matched
    return result

if __name__ == "__main__":
    for idx, row in data.iterrows():
        candidates = row.clip_baseline
        candidates = [item[0] for item in candidates]
        for image_id in candidates:
            if not os.path.exists(SAVE_MATCHING_PATH.format(idx, image_id)):
                try:
                    image_list = get_dense_caption(image_id)
                    object_list = get_sentence_json(idx)
                    match = generate_caption_to_image(row.sentence, object_list, image_list)
                    with open(SAVE_MATCHING_PATH.format(idx, image_id), 'w') as f:
                        json.dump(match, f)
                except JSONDecodeError as e:
                    print("failed {} {}".format(idx, image_id))
                except FileNotFoundError as e:
                    print("failed {} {}".format(idx, image_id))
