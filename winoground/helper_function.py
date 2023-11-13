import json
from PIL import Image, ImageDraw
import numpy as np

def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

def scoreing(scores):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(scores)
    print(text_correct_count, image_correct_count, group_correct_count)
    print("text score:", text_correct_count/denominator)
    print("image score:", image_correct_count/denominator)
    print("group score:", group_correct_count/denominator)

def send_gpu(original_processor, device):
    original_processor['pixel_values'] = original_processor['pixel_values'].to(device)
    original_processor['attention_mask'] = original_processor['attention_mask'].to(device)
    original_processor['input_ids'] = original_processor['input_ids'].to(device)
    return original_processor

def get_matching(row_id, caption_id, image_id):
    f = open("match_relation/{}_caption{}_image{}.json".format(row_id, caption_id, image_id))
    return json.load(f)

def get_sentence_json(row_id, caption_id, relatin_path):
    f = open(relatin_path.format(row_id, caption_id))
    result = json.load(f)
    return json.loads(result)

def black_outside_rectangle(image, left_top, right_bottom):
    blacked_out_image = Image.new("RGB", image.size, color="black")
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left_top, right_bottom], fill=255)
    blacked_out_image.paste(image, mask=mask)
    return blacked_out_image

def create_sub_image_obj(row_id, caption_id, image_id, image_path, caption_path, relation_path):
    object_image = {}
    old_key_to_new_key = {}
    matched_objects = get_matching(row_id, caption_id, image_id)
    image = Image.open(image_path.format(row_id, image_id))
    f = open(caption_path.format(row_id, image_id))
    attributes = open(relation_path.format(row_id, caption_id))
    attributes = json.loads(json.load(attributes))["objects"]
    location = json.load(f)
    for key, object_name in matched_objects.items():
        if key in attributes and "attributes" in attributes[key]:
            key_name = key
            if attributes[key]["attributes"] and type(attributes[key]["attributes"]) == str:
                key_name = "{} {}".format(attributes[key]["attributes"], key)
            elif attributes[key]["attributes"] and type(attributes[key]["attributes"]) == list and len(attributes[key]["attributes"]) > 0:
                key_name = "{} {}".format(" ".join(attributes[key]["attributes"]), key)
            else:
                key_name = key
        if len(object_name) == 0:
            object_image[key_name] = image
        else:
            subimages = []
            for object in object_name:
                if object in location:
                    left_top_x, left_top_y, right_bottom_x, right_bottom_y = location[object][0]
                    blacked_out_image = black_outside_rectangle(image, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y))
                    subimages.append(blacked_out_image)
                else:
                    subimages.append(image)
            object_image[key_name] = overlay_images(subimages)
        old_key_to_new_key[key] = key_name
    return object_image, old_key_to_new_key

def overlay_images(images):
    base_image = Image.new("RGB", images[0].size, (0, 0, 0))
    mask = Image.new("L", base_image.size, 0)
    for image in images:
        image = image.convert("RGBA")
        image_mask = image.convert("L")
        image_mask = image_mask.point(lambda x: 255 if x > 0 else 0, mode="1")
        base_image.paste(image, (0, 0), mask=image_mask)
    return base_image

def normalize_tensor_list(tensor_list):
    total_sum = sum(tensor.item() for tensor in tensor_list)
    normalized_list = [tensor / total_sum for tensor in tensor_list]
    return normalized_list

def create_relation_object(object_images, text_json, row_id, image_id, key_map, image_path):
    if "connections" not in text_json:
        return None, None
    relations = text_json["connections"]
    object_names = list(object_images.keys())
    key_names = list(key_map.keys())
    if type(relations) is not list or len(relations) == 0:
        return None, None
    verb_images = []
    verbs = []
    for relation in relations:
        if "subject" in relation:
            subject = relation["subject"]
        else:
            subject = None
        if "verb" in relation:
            verb = relation["verb"]
        else:
            verb = None
        if "object" in relation:
            object = relation["object"]
        else:
            object = None
        if subject not in key_names and object in key_names:
            verb_image = Image.open(image_path .format(row_id, image_id))
        elif subject in key_names and object not in key_names:
            verb_image = Image.open(image_path .format(row_id, image_id))
        elif subject in key_names and object in key_names:
            verb_image = overlay_images([object_images[key_map[object]], object_images[key_map[subject]]])
        else:
            verb_image = Image.open(image_path.format(row_id, image_id))
        verb_images.append(verb_image)
        verbs.append(verb)
    return verb_images, verbs