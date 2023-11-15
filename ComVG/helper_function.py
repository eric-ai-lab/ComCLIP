import json
from PIL import Image, ImageDraw
import numpy as np

def read_image(id, path_format):
    return Image.open(path_format.format(id))

def get_matching(caption_id, image_id, matched_path):
    f = open(matched_path.format(caption_id, image_id))
    result = json.load(f)
    return result    

def black_outside_rectangle(image, left_top, right_bottom):
    blacked_out_image = Image.new("RGB", image.size, color="black")
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left_top, right_bottom], fill=255)
    blacked_out_image.paste(image, mask=mask)
    return blacked_out_image

def create_sub_image_obj(sentence_id, image_id, image_path, densecaption_path, matched_path):
    print(sentence_id, image_id, image_path, densecaption_path, matched_path)
    object_image = {}
    matched_objects = get_matching(sentence_id, image_id, matched_path)
    image = Image.open(image_path.format(image_id))
    location = json.load(open(densecaption_path.format(image_id)))
    for key, object_name in matched_objects.items():
        if len(object_name) == 0:
            object_image[key] = image
        else:
            subimages = []
            for object in object_name:
                if object in location:
                    left_top_x, left_top_y, right_bottom_x, right_bottom_y = location[object][0]
                    blacked_out_image = black_outside_rectangle(image, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y))
                    subimages.append(blacked_out_image)
                else:
                    subimages.append(image)
            object_image[key] = overlay_images(subimages)
    return object_image, matched_objects

def overlay_images(images):
    base_image = Image.new("RGB", images[0].size, (0, 0, 0))
    mask = Image.new("L", base_image.size, 0)
    for image in images:
        image = image.convert("RGBA")
        image_mask = image.convert("L")
        image_mask = image_mask.point(lambda x: 255 if x > 0 else 0, mode="1")
        base_image.paste(image, (0, 0), mask=image_mask)
    return base_image

def create_relation_object(object_images, subj, relation, obj, image_id, matched_json, image_path):
    relations = [relation]
    if type(relations) is not list or len(relations) == 0:
        return None, None
    verb_images = []
    verbs = []
    for relation in relations:
        if len(matched_json[subj]) == 0 and len(matched_json[obj]) != 0:
            verb_image = Image.open(image_path.format(image_id))
        elif len(matched_json[subj]) != 0 and len(matched_json[obj]) == 0:
            verb_image = Image.open(image_path.format(image_id))
        elif len(matched_json[subj]) != 0 and len(matched_json[obj]) != 0:
            verb_image = overlay_images([object_images[obj], object_images[subj]])
        else:
            verb_image = Image.open(image_path.format(image_id))
        verb_images.append(verb_image)
        verbs.append(relation)
    return verb_images, verbs

def normalize_tensor_list(tensor_list):
    total_sum = sum(tensor.item() for tensor in tensor_list)
    normalized_list = [tensor / total_sum for tensor in tensor_list]
    return normalized_list

