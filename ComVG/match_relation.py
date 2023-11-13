import pandas as pd
import os
import openai
import json
import argparse

parser = argparse.ArgumentParser(description='Match SVO to Dense Caption.')
parser.add_argument("--dataset_path", type=str, help='path to the data csv')
parser.add_argument("--densecaption_path", type=str, help='folder for dense caption')
parser.add_argument("--openai", type=str, help='openai api key')
args = parser.parse_args()
openai.api_key = args.openai

data = pd.read_csv(args.dataset_path)
densecaption_path = args.densecaption_path
matched_json_file_save_path = "matched_realtion"


def get_dense_caption(image_id):
    f = open(densecaption_path.format(image_id))
    result = list(json.load(f).keys())
    return result

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
    for obj in obejct_list:
        result[obj] = match_objects(caption, obj, image_list)["labels"]
    return result

def match_caption_to_entitiy():
    for idx, row in data.iterrows():
        triplet = row.pos_triplet.replace("[", "")
        triplet = triplet.replace("]", "")
        svo = triplet.split(",")
        subject, obj = svo[0], svo[-1]
        object_list = [subject, obj]
        if not os.path.exists(matched_json_file_save_path.format(row.sentence_id, row.pos_image_id)):
            with open(matched_json_file_save_path.format(row.sentence_id, row.pos_image_id), 'w') as f:
                pos_image_list = get_dense_caption(row.pos_image_id)
                pos_result = generate_caption_to_image(row.sentence, object_list, pos_image_list)
                json.dump(pos_result, f)
        if not os.path.exists(matched_json_file_save_path.format(row.sentence_id, row.neg_image_id)):
            with open(matched_json_file_save_path.format(row.sentence_id, row.neg_image_id), 'w') as f:
                neg_image_list = get_dense_caption(row.neg_image_id)
                neg_result = generate_caption_to_image(row.sentence, object_list, neg_image_list)
                json.dump(neg_result, f)

if __name__ == "__main__":
    match_caption_to_entitiy()
    