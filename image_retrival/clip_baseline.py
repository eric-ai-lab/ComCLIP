from helper_function import *
import pandas as pd
import torch
import clip
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type = str, help="csv file for the flickr30k")
parser.add_argument("model", type=str, help="RN50, ViT/B-32, ViT/L-14")
args = parser.parse_args()

device = "cuda:0"
data = pd.read_pickle(args.dataset)
model, preprocess = clip.load(args.model, device='cpu')
model.cuda(device).eval()

def clip_compute_one_pair(caption, image_id):
    image = preprocess(read_image(image_id))
    text_input = clip.tokenize(caption).cuda(device)
    image_input = torch.tensor(np.stack([image])).cuda(device)
    with torch.no_grad():
        original_image_embed = model.encode_image(image_input).float()
        original_text_embed = model.encode_text(text_input).float()
    image_features = original_image_embed / original_image_embed.norm(dim=-1, keepdim=True).float()
    text_features = original_text_embed /original_text_embed.norm(dim=-1, keepdim=True).float()
    similarity = text_features.detach().cpu().numpy() @ image_features.detach().cpu().numpy().T
    return similarity

def get_score(row_id):
    result = {}
    row = data.iloc[row_id]
    for candidate in range(1000):
        result[candidate] = clip_compute_one_pair(row_id, row.sentence, candidate).item()
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    return result

if __name__ == "__main__":
    clip_score = {}
    for idx, row in data.iterrows():
        try:
            clip_score[idx] = get_score(idx)
        except Exception as e:
            continue
    top_1 = 0
    top_5 = 0
    top_10 = 0
    hit_1_comclip = []
    for idx, value in comclip.items():
        candidates = list(value.keys())
        candidates = [int(i) for i in candidates]
        if candidates[0] == int(idx):
            top_1 += 1
            hit_1_comclip.append(int(idx))
        if int(idx) in candidates[:5]:
            top_5 += 1
        if int(idx) in candidates:
            top_10 += 1
    top_1/ len(mscoco_1k_test), top_5/ len(mscoco_1k_test), top_10/ len(mscoco_1k_test)
    
    