import torch
import argparse
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.transforms import Resize, Normalize
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import numpy as np
import models
from tokenizer import SimpleTokenizer
from tqdm import tqdm
import json
import pandas as pd
from helper_function import * 
from datasets import load_dataset
parser = argparse.ArgumentParser(description='Test SLIP on winoground.')
parser.add_argument('--RelationPath', type=str, help='Path to text relation files.')
parser.add_argument('--CaptionPath', type=str, help='Path store densecaption.')
parser.add_argument('--huggingface_token', type=str, help='Huggingface token from the Hugging Face account.')
parser.add_argument('--slip_weigth_path', type=str, help='path to slip weights')
parser.add_argument('--image_path', type=str, help='path to slip weights')
args = parser.parse_args()

ckpt = torch.load(args.slip_weigth_path, map_location='cpu')
image_path = args.image_path+"/ex_{}_img_{}.png"
relation_path = args.RelationPath+"/{}_{}.json"
caption_path = args.CaptionPath+"/ex_{}_img_{}.json"
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    state_dict[k.replace('module.', '')] = v
old_args = ckpt['args']
print("=> creating model: {}".format(old_args.model))
model = getattr(models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
model.load_state_dict(state_dict, strict=True)
model.cuda('cuda:0')

winoground = load_dataset("facebook/winoground", use_auth_token=args.huggingface_token)["test"]

tokenizer = SimpleTokenizer()
transform = transforms.Compose([Resize((224,224), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                                transforms.ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                ])

def subimage_score_embedding(image, text):
    if text:
        image_embed = model.encode_image(transform(image).cuda("cuda:0").unsqueeze(0))
        text_embed = model.encode_text(tokenizer(text).cuda("cuda:0").view(-1, 77).contiguous())
        score = F.cosine_similarity(image_embed, text_embed)
        return image_embed, score
    else:
        return None, None
    
def comclip_one_pair(row_id, caption_id, image_id):
    if caption_id == 0:
        caption = winoground[row_id]["caption_0"]
    else:
        caption = winoground[row_id]["caption_1"]
    if image_id == 0:
        image = winoground[row_id]["image_0"].convert("RGB")
    else:
        image = winoground[row_id]["image_1"].convert("RGB")
    original_image_embed = model.encode_image(transform(image).cuda("cuda:0").unsqueeze(0))
    original_text_embed = model.encode_text(tokenizer(caption ).cuda("cuda:0").view(-1, 77).contiguous())
    text_json = get_sentence_json(row_id, caption_id)
    object_images, key_map = create_sub_image_obj(row_id, caption_id, image_id, image_path, caption_path, relation_path)
    relation_images, relation_words = create_relation_object(object_images, text_json, row_id, image_id, key_map, image_path)
    if relation_images and relation_words:
        for relation_image, word in zip(relation_images, relation_words):
            object_images[word] = relation_image

    ##subimages
    image_embeds = []
    image_scores = []
    for key, sub_image in object_images.items():
        # print(key)
        image_embed, image_score = subimage_score_embedding(sub_image, key)
        if image_embed is not None and image_score is not None:
            image_embeds.append(image_embed)
            image_scores.append(image_score)
    #regularize the scores
    similarity = normalize_tensor_list(image_scores)
    for score, image in zip(similarity, image_embeds):
        # print(score)
        original_image_embed += score * image
    image_features = original_image_embed / original_image_embed.norm(dim=-1, keepdim=True).float()
    text_features = original_text_embed /original_text_embed.norm(dim=-1, keepdim=True).float()
    similarity = text_features.detach().cpu().numpy() @ image_features.detach().cpu().numpy().T
    return similarity, object_images, relation_images
    
def get_score(id):
    result = {}
    result["id"] = id
    result["c0_i0"] = comclip_one_pair(id, 0, 0)[0].item()
    result["c0_i1"] = comclip_one_pair(id, 0, 1)[0].item()
    result["c1_i0"] = comclip_one_pair(id, 1, 0)[0].item()
    result["c1_i1"] = comclip_one_pair(id, 1, 1)[0].item()
    return result

if __name__ == "__main__":
    comslip_score = []
    for i in range(400):
        try:
            comslip_score.append(get_score(i))
        except Exception as e:
            continue
    scoreing(comslip_score)