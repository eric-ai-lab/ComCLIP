from datasets import load_dataset
from helper_function import *
import argparse
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Test BLIP image-text matching on winoground.')
parser.add_argument('--RelationPath', type=str, help='Path to text relation files.')
parser.add_argument('--CaptionPath', type=str, help='Path store densecaption.')
parser.add_argument('--image_path', type=str, help='Path that stores image.')
parser.add_argument('--huggingface_token', type=str, help='Path that stores image.')
args = parser.parse_args()

relation_path = args.RelationPath+"/{}_{}.json"
caption_path = args.CaptionPath+"/ex_{}_img_{}.json"
image_path = args.image_path+"/ex_{}_img_{}.png"
auth_token = args.huggingface_token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
model.to(device)

def subimage_score_embedding_blip(image, text):
    if text:
        image = vis_processors["eval"](image.convert("RGB")).unsqueeze(0).to(device)
        text = txt_processors["eval"](text)
        sample = {"image": image, "text_input": [text]}
        features_image = model.extract_features(sample, mode="image")
        features_text = model.extract_features(sample, mode="text")
        similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
        return features_image.image_embeds, similarity
    else:
        return None, None
    
def inference_one_pair(row_id, text_id, image_id):
    caption = winoground[row_id]["caption_{}".format(text_id)]
    image = winoground[row_id]["image_{}".format(image_id)].convert("RGB")
    original_image = vis_processors["eval"](image).unsqueeze(0).to(device)
    text = txt_processors["eval"](caption)
    sample = {"image": original_image, "text_input": [text]}
    original_image_embed = model.extract_features(sample, mode="image").image_embeds
    text_features = model.extract_features(sample, mode="text").text_embeds_proj[:,0,:].t()

    text_json = get_sentence_json(row_id, text_id)
    object_images, key_map = create_sub_image_obj(row_id, text_id, image_id, image_path, caption_path, relation_path)
    relation_images, relation_words = create_relation_object(object_images, text_json, row_id, image_id, key_map, image_path)
    if relation_images and relation_words:
        for relation_image, word in zip(relation_images, relation_words):
            object_images[word] = relation_image

    ##subimages
    image_embeds = []
    image_scores = []
    for key, sub_image in object_images.items():
        image_embed, image_score = subimage_score_embedding_blip(sub_image, key)
        if image_embed is not None and image_score is not None:
            image_embeds.append(image_embed)
            image_scores.append(image_score)
    #regularize the scores
    similarity = normalize_tensor_list(image_scores)
    for score, image in zip(similarity, image_embeds):
        original_image_embed += score * image
    image_features = original_image_embed / original_image_embed.norm(dim=-1, keepdim=True).float()
    similarity = (F.normalize(model.vision_proj(image_features), dim=-1) @ text_features).max()
    return similarity

def get_score(id):
    result = {}
    result["id"] = id
    result["c0_i0"] = inference_one_pair(id, 0, 0).item()
    result["c0_i1"] = inference_one_pair(id, 0, 1).item()
    result["c1_i0"] = inference_one_pair(id, 1, 0).item()
    result["c1_i1"] = inference_one_pair(id, 1, 1).item()
    return result

def inference(winoground):
    comblip_score = []
    for i in range(len(winoground)):
        try:
            comblip_score.append(get_score(i))
        except Exception as e:
            continue
    print(scoreing(comblip_score))

if __name__ == "__main__":
    inference(winoground)

