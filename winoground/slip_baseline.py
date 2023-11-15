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
parser.add_argument('--huggingface_token', type=str, help='Huggingface token from the Hugging Face account.')
parser.add_argument('--slip_weigth_path', type=str, help='path to slip weights')
args = parser.parse_args()

ckpt = torch.load(args.slip_weight_path, map_location='cpu')
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

if __name__ == "__main__":
    winoground_slip_scores = []
    for example in tqdm(winoground):
        image_0 = transform(example["image_0"].convert("RGB")).cuda("cuda:2").unsqueeze(0)
        image_1 = transform(example["image_1"].convert("RGB")).cuda("cuda:2").unsqueeze(0)
        text_0 = tokenizer(example["caption_0"]).cuda("cuda:2").view(-1, 77).contiguous()
        text_1 = tokenizer(example["caption_1"]).cuda("cuda:2").view(-1, 77).contiguous()
        text_0_embed = model.encode_text(text_0)
        text_1_embed = model.encode_text(text_1)
        image_0_embed = model.encode_image(image_0)
        image_1_embed = model.encode_image(image_1)

        c0_i0 = F.cosine_similarity(image_0_embed,text_0_embed).item()
        c1_i0 = F.cosine_similarity(image_0_embed,text_1_embed).item()
        c0_i1 = F.cosine_similarity(image_1_embed,text_0_embed).item()
        c1_i1 = F.cosine_similarity(image_1_embed,text_1_embed).item()
        winoground_slip_scores.append({"id" : example["id"], "c0_i0": c0_i0, "c0_i1": c0_i1, "c1_i0": c1_i0, "c1_i1": c1_i1})
    print(scoreing(winoground_slip_scores))