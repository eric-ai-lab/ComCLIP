from datasets import load_dataset
from helper_function import *
import argparse
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

parser = argparse.ArgumentParser(description='Test CLIP ViT/B-32 on winoground.')
parser.add_argument('--huggingface_token', type=str, help='Huggingface token from the Hugging Face account.')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(args):
    auth_token = args.huggingface_token
    winoground = load_dataset("facebook/winoground", token=auth_token)["test"]
    return winoground

def load_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    return clip_model, clip_processor

def inference(clip_model, clip_processor, winoground):
    winoground_clip_scores = []
    for example in tqdm(winoground):
        input_c0_i0 = send_gpu(clip_processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")], return_tensors="pt"), device)
        input_c1_i0 = send_gpu(clip_processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")], return_tensors="pt"), device)
        input_c0_i1 = send_gpu(clip_processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")], return_tensors="pt"), device)
        input_c1_i1 = send_gpu(clip_processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")], return_tensors="pt"), device)
        output_c0_i0 = clip_model(**input_c0_i0)
        output_c1_i0 = clip_model(**input_c1_i0)
        output_c0_i1 = clip_model(**input_c0_i1)
        output_c1_i1 = clip_model(**input_c1_i1)
        clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
        clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
        clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
        clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
        winoground_clip_scores.append({"id": example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1})
    scores = scoreing(winoground_clip_scores)
    print(scores)

if __name__ == "__main__":
    winoground = load_data(args)
    print("Data loaded...")
    clip, clip_processor = load_model()
    print("Model loaded...")
    inference(clip, clip_processor, winoground)



    

    