import torch
import argparse
from helper_function import *
import clip
import torch
import pandas as pd
GPU=0
parser = argparse.ArgumentParser(description='Test CLIP on ComVG.')
parser.add_argument('--model', type=str, help='Your model verion: RN50, ViT/B-32, ViT/L-14')
parser.add_argument("--data_path", type=str, help='path to comVG.csv')
parser.add_argument('--image_path', type=str, help='path to images')
args = parser.parse_args()
model, preprocess = clip.load(args.model, device='cpu')
model.cuda(GPU).eval()

IMAGE_PATH = args.image_path + "/{}.jpg" 
data = pd.read_csv(args.data_path)
data.head(1)

def inference():
    clip_scores = []
    for idx, row in data.iterrows():
        text_tokens = clip.tokenize([row.sentence]).cuda(GPU)
        image_pos, image_neg = preprocess(read_image(row.pos_image_id, IMAGE_PATH)), preprocess(read_image(row.neg_image_id, IMAGE_PATH))
        images = [image_pos, image_neg]
        image_input = torch.tensor(np.stack(images)).cuda(GPU)
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        clip_scores.append({"id":idx, "pos_score": similarity[0][0], "neg_score": similarity[0][1]})
    return clip_scores

def report_score(clip_scores):
    acc = 0
    for i in clip_scores:
        if i["pos_score"] > i["neg_score"]:
            acc += 1
    return acc / len(clip_scores)

if __name__ == "__main__":
    report_score(inference())