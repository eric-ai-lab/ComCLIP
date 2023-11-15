from datasets import load_dataset
from helper_function import *
import argparse
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
device = "cuda:0"
parser = argparse.ArgumentParser(description='Test BILP2 on winoground.')
parser.add_argument('--huggingface_token', type=str, help='Huggingface token from the Hugging Face account.')
parser.add_argument('--image_path', type=str, help='images of all winoground dataset')
args = parser.parse_args()
IMAGE_PATH = args.image_path+"/ex_{}_img_{}.png"

auth_token = args.huggingface_token
winoground = load_dataset("facebook/winoground", token=auth_token)["test"]

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

def inference(model, vis_processors, text_processors, winoground):
    winoground_blip_scores = []
    for i in tqdm(range(len(winoground))):
        image_0 = Image.open(IMAGE_PATH.format(i, 0))
        image_0 = vis_processors["eval"](image_0.convert("RGB")).unsqueeze(0).to(device)
        image_1 = Image.open(IMAGE_PATH.format(i, 1))
        image_1 = vis_processors["eval"](image_1.convert("RGB")).unsqueeze(0).to(device)
        text_0 = text_processors["eval"](winoground[i]["caption_0"])
        text_1 = text_processors["eval"](winoground[i]["caption_1"])
        c0_i0 = model({"image": image_0, "text_input": text_0}, match_head='itc')[0].item()
        c0_i1 = model({"image": image_1, "text_input": text_0}, match_head='itc')[0].item()
        c1_i0 = model({"image": image_0, "text_input": text_1}, match_head='itc')[0].item()
        c1_i1 = model({"image": image_1, "text_input": text_1}, match_head='itc')[0].item()
        winoground_blip_scores.append({"id" : i, "c0_i0": c0_i0, "c0_i1": c0_i1, "c1_i0": c1_i0, "c1_i1": c1_i1})
    scores = scoreing(winoground_blip_scores)
    print(scores)

if __name__ == "__main__":
    print(inference(model, vis_processors, text_processors, winoground))
