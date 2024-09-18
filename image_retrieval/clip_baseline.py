from helper_function import *
import pandas as pd
import torch
import clip
import argparse
import Image
from transformers import BLIP2Processor, BLIP2Model, AutoTokenizer
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, help="csv file for the flickr30k")
parser.add_argument("--model", type=str, help="Salesforce/blip2-opt-2.7b")
parser.add_argument("--image_path", type=str, help="path to the image")
args = parser.parse_args()

device = "cuda:0"
data = pd.read_pickle(args.dataset)
preprocess= BLIP2Processor.from_pretrained(args.model,device='cpu')
model = BLIP2Model.from_pretrained(args.model,device='cpu')
tokenizer = AutoTokenizer.from_pretrained(args.model,device='cpu')
#model, preprocess = clip.load(args.model, device='cpu')
model.cuda(device).eval()
IMAGE_PATH = args.image_path+ "/{}.jpg"

def clip_compute_one_pair(caption, image_id):

    image = Image.open(image_id)
    image_inputs = processor(images=image, return_tensors="pt")
    image_input = model.get_image_features(**image_inputs).cuda(device)

    image = preprocess(read_image(image_id, IMAGE_PATH))
    text_inputs = tokenizer(caption, padding=True, return_tensors="pt")
    text_input= = model.get_text_features(**text_inputs).cuda(device)
    #text_input = clip.tokenize(caption).cuda(device)
    #image_input = torch.tensor(np.stack([image])).cuda(device)
    with torch.no_grad():
        original_image_embed = image_input.float()
        original_text_embed =  text_input.float()
        #original_image_embed = model.encode_image(image_input).float()
        #original_text_embed = model.encode_text(text_input).float()
    image_features = original_image_embed / original_image_embed.norm(dim=-1, keepdim=True).float()
    text_features = original_text_embed /original_text_embed.norm(dim=-1, keepdim=True).float()
    similarity = text_features.detach().cpu().numpy() @ image_features.detach().cpu().numpy().T
    return similarity

def get_score(row_id):
    result = {}
    row = data.iloc[row_id]
    for candidate in range(1000):
        result[candidate] = clip_compute_one_pair(row.sentence, candidate).item()
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    print(result)
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
    for idx, value in clip_score.items():
        candidates = list(value.keys())
        candidates = [int(i) for i in candidates]
        if candidates[0] == int(idx):
            top_1 += 1
        if int(idx) in candidates[:5]:
            top_5 += 1
        if int(idx) in candidates:
            top_10 += 1
    print("Top 1 {}. Top 5 {}".format(top_1/ 1000, top_5/ 1000))
    
    
