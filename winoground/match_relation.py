from datasets import load_dataset
import openai
import json
from json import JSONDecodeError
import time
import argparse
parser = argparse.ArgumentParser(description='Match text to densecaption on winoground.')
parser.add_argument('--huggingface_token', type=str, help='Huggingface token from the Hugging Face account.')
parser.add_argument('--openai', type=str, help='OpenAI api key.')
parser.add_argument("--RelationPath", type=str, help="Folder path that stores dense caption from GRiT")
parser.add_argument("--DenseCaptionPath", type=str, help="Folder path that stores dense caption from GRiT")
args = parser.parse_args()
openai.api_key = args.openai
CAPTION_PATH = args.DenseCaptionPath + "/ex_{}_img_{}.json"
RELATION_JSON_PATH = args.RelationPath + "/{}_{}.json"
SAVE_MATCHING_PATH = 'matched_relation/{}_caption{}_image{}.json'

def load_preset(args):
    auth_token = args.huggingface_token
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]
    return winoground

def get_dense_caption(id, image_id):
    f = open(CAPTION_PATH.format(id, image_id))
    result = list(json.load(f).keys())
    return result
def get_sentence_json(row_id, caption_id):
    f = open(RELATION_JSON_PATH.format(row_id, caption_id))
    result = json.load(f)
    return list(json.loads(result)["objects"].keys())

def match_objects(sentence, object, list):
    try:
        input = "Given a list of labels in triple quotes \"\"\"{}\"\"\" ".format(list) + \
                "and a sentence in triple quotes \"\"\"{}\"\"\". Tell me which labels in the list could refer to ".format(sentence) + \
                "the \"{}\" from the sentence. Only output the results in json format with 1 key named labels. If none of the labels refer to the \"{}\", set the value in the json to be an empty list".format(object, object)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}], temperature=0)
        return json.loads(completion["choices"][0]["message"]["content"])
    except Exception as e:
        return {"labels":[]}
    
def generate_caption_to_image(caption, obejct_list, image_list):
    result = {}
    for object in obejct_list:
        result[object] = match_objects(caption, object, image_list)["labels"]
    return result

def match_caption_to_relation(winoground):
    for i in range(len(winoground)):
        caption_0 = winoground[i]["caption_0"]
        caption_1 = winoground[i]["caption_1"]
        caption_0_objects = get_sentence_json(i, 0)
        caption_1_objects = get_sentence_json(i, 1)
        image_0_list = get_dense_caption(i, 0)
        image_1_list = get_dense_caption(i, 1)
        with open(SAVE_MATCHING_PATH.format(i, 0, 0), 'w') as f:
            caption_0_image_0 = generate_caption_to_image(caption_0, caption_0_objects, image_0_list)
            json.dump(caption_0_image_0, f)
        with open(SAVE_MATCHING_PATH.format(i, 0, 1), 'w') as f:
            caption_0_image_1 = generate_caption_to_image(caption_0, caption_0_objects, image_1_list)
            json.dump(caption_0_image_1, f)
        with open(SAVE_MATCHING_PATH.format(i, 1, 0), 'w') as f:
            caption_1_image_0 = generate_caption_to_image(caption_1, caption_1_objects, image_0_list)
            json.dump(caption_1_image_0, f)
        with open(SAVE_MATCHING_PATH.format(i, 1, 1), 'w') as f:
            caption_1_image_1 = generate_caption_to_image(caption_1, caption_1_objects, image_1_list)
            json.dump(caption_1_image_1, f)
        time.sleep(1)

if __name__ == "__main__":
    winoground = load_preset(args)
    match_caption_to_relation(winoground)