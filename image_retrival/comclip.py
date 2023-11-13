from helper_function import *
import pandas as pd
import torch
import clip
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type = str)
parser.add_argument("model", type=str)
args = parser.parse_args()

device = "cuda:0"
data = pd.read_pickle("dataset.pkl") ### Flickr30k or MSCOCO test set
model, preprocess = clip.load("RN50", device='cpu')
model.cuda(device).eval()

def subimage_score_embedding(image, text):
    if text:
        image = preprocess(image)
        text_input = clip.tokenize(text).cuda(device)
        image_input = torch.tensor(np.stack([image])).cuda(device)
        with torch.no_grad():
            image_embed = model.encode_image(image_input).float()
            text_embed = model.encode_text(text_input).float()
        score = text_embed @ image_embed.T
        return image_embed, score
    else:
        return None, None
    
def comclip_one_pair(row_id, caption, image_id):
    image = preprocess(read_image(image_id))
    text_input = clip.tokenize(caption).cuda(device)
    image_input = torch.tensor(np.stack([image])).cuda(device)
    with torch.no_grad():
        original_image_embed = model.encode_image(image_input).float()
        original_text_embed = model.encode_text(text_input).float()
    text_json = get_sentence_json(row_id)
    object_images, key_map = create_sub_image_obj(row_id, image_id)
    relation_images, relation_words = create_relation_object(object_images, text_json, row_id, image_id, key_map)
    if relation_images and relation_words:
        for relation_image, word in zip(relation_images, relation_words):
            if word in object_images:
                object_images[word+"_dup"] = relation_image
            else:
                object_images[word] = relation_image

    ##subimages
    image_embeds = []
    image_scores = []
    for key, sub_image in object_images.items():
        if "_dup" in key:
            key = key.replace("_dup", "")
        image_embed, image_score = subimage_score_embedding(sub_image, key)
        if image_embed is not None and image_score is not None:
            image_embeds.append(image_embed)
            image_scores.append(image_score)
    #regularize the scores
    similarity = normalize_tensor_list(image_scores)
    for score, image in zip(similarity, image_embeds):
        original_image_embed += score * image
    image_features = original_image_embed / original_image_embed.norm(dim=-1, keepdim=True).float()
    text_features = original_text_embed /original_text_embed.norm(dim=-1, keepdim=True).float()
    similarity = text_features.detach().cpu().numpy() @ image_features.detach().cpu().numpy().T
    return similarity

def get_score(row_id):
    result = {}
    row = data.iloc[row_id]
    candidates = row.clip_top_ten_pick
    for candidate in candidates:
        result[candidate[0]] = comclip_one_pair(row_id, row.sentence, candidate[0]).item()
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    return result

if __name__ == "__main__":
    comclip_score = {}
    for idx, row in data.iterrows():
        try:
            comclip_score[idx] = get_score(idx)
        except Exception as e:
            print(e)
    top_1 = 0
    top_5 = 0
    for idx, value in comclip_score.items():
        candidates = list(value.keys())
        candidates = [int(i) for i in candidates]
        if candidates[0] == int(idx):
            top_1 += 1
        if int(idx) in candidates[:5]:
            top_5 += 1
    print("Top 1 score: {}. Top 5 score: {}".format(top_1/ len(1000), top_5/ len(1000)))