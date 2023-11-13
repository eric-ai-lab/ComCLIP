from datasets import load_dataset
from helper_function import *
import argparse
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

parser = argparse.ArgumentParser(description='Test CLIP ViT/B-32 on winoground.')
parser.add_argument('--RelationPath', type=str, help='Path to text relation files.')
args = parser.parse_args()
relation_path = args.RelationPath+"{}_{}.json"
auth_token = args.huggingface_token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)

def subimage_score_embedding(image, text, clip_processor, clip_model):
    if text:
        processor = send_gpu(clip_processor(text=[text], images=[image], return_tensors="pt"), device)
        output = clip_model(**processor)
        image_embed = output.image_embeds
        score = output.logits_per_text
        return image_embed, score
    else:
        return None, None
    
def inference_one_pair(row_id, caption_id, image_id):
    if caption_id == 0:
        caption = winoground[row_id]["caption_0"]
    else:
        caption = winoground[row_id]["caption_1"]
    if image_id == 0:
        image = winoground[row_id]["image_0"].convert("RGB")
    else:
        image = winoground[row_id]["image_1"].convert("RGB")
    original_processor = send_gpu(clip_processor(text=[caption], images=[image], return_tensors="pt"), "cuda:0")
    original_output = clip_model(**original_processor)
    original_image_embed = original_output.image_embeds
    original_text_embed = original_output.text_embeds
    text_json = get_sentence_json(row_id, caption_id, relation_path)
    object_images, key_map = create_sub_image_obj(row_id, caption_id, image_id)
    relation_images, relation_words = create_relation_object(object_images, text_json, row_id, image_id, key_map)
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

def get_score(id):
    result = {}
    result["id"] = id
    result["c0_i0"] = inference_one_pair(id, 0, 0).item()
    result["c0_i1"] = inference_one_pair(id, 0, 1).item()
    result["c1_i0"] = inference_one_pair(id, 1, 0).item()
    result["c1_i1"] = inference_one_pair(id, 1, 1).item()
    return result

def inference(winoground):
    comclip_score = []
    for i in range(len(winoground)):
        try:
            comclip_score.append(get_score(i))
        except Exception as e:
            continue
    print(scoreing(comclip_score))

if __name__ == "__main__":
    inference(winoground)

