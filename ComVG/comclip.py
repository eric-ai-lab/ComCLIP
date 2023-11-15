import torch
from helper_function import *
import clip
import torch
import argparse
import pandas as pd
GPU = 0

parser = argparse.ArgumentParser(description='Match SVO to Dense Caption.')
parser.add_argument("--dataset_path", type=str, help='path to the data csv')
parser.add_argument("--image_path", type=str, help='path to the image')
parser.add_argument("--densecaption_path", type=str, help='folder for dense caption')
parser.add_argument("--model", type=str, help='RN50, ViT/B-32, ViT/L-14')
args = parser.parse_args()
IMAGE_PATH = args.image_path + "/{}.jpg" 

data = pd.read_csv(args.data_path)
model, preprocess = clip.load("RN50", device='cpu')
model.cuda(GPU).eval()

def subimage_score_embedding(image, text):
    if text:
        image = preprocess(image)
        text_input = clip.tokenize(text).cuda(GPU)
        image_input = torch.tensor(np.stack([image])).cuda(GPU)
        with torch.no_grad():
            image_embed = model.encode_image(image_input).float()
            text_embed = model.encode_text(text_input).float()
        score = text_embed @ image_embed.T
        return image_embed, score
    else:
        return None, None
    
def inference_one_pair(row, caption, image_id):
    image = preprocess(read_image(image_id, IMAGE_PATH))
    text_input = clip.tokenize(row.sentence).cuda(GPU)
    image_input = torch.tensor(np.stack([image])).cuda(GPU)
    with torch.no_grad():
        original_image_embed = model.encode_image(image_input).float()
        original_text_embed = model.encode_text(text_input).float()

    svo = row.pos_triplet.split(",")
    subj, verb, obj = svo[0], svo[1], svo[-1]
    object_images, matched_json = create_sub_image_obj(row.sentence_id, image_id, IMAGE_PATH, CAPTION_PATH, MATCHING_PATH)
    relation_images, relation_words = create_relation_object(object_images, subj, verb, obj, image_id, matched_json, IMAGE_PATH)
    if relation_images and relation_words:
        for relation_image, word in zip(relation_images, relation_words):
            object_images[word] = relation_image

    ##subimages
    image_embeds = []
    image_scores = []
    for key, sub_image in object_images.items():
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

def compute_one_row(idx, row):
    result_pos = inference_one_pair(row, row.sentence, row.pos_image_id).item()
    result_neg = inference_one_pair(row, row.sentence, row.neg_image_id).item()
    result = {"id" : idx, "pos_score": result_pos, "neg_score": result_neg}
    return result 

if __name__ == "__main__":
    comclip_score = []
    for idx, row in data.iterrows():
        try:
            score = compute_one_row(idx, row)
            comclip_score.append(score)
        except Exception as e:
            continue
    acc = 0
    for i in comclip_score:
        if i["pos_score"] > i["neg_score"]:
            acc += 1
    print(acc / len(comclip_score))