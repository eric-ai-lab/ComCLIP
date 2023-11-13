import json
from PIL import Image, ImageDraw
import numpy as np
IMAGE_PATH = "path/to/image/{}.jpg"
MATCHING_PATH = "../comvg-matching/sentence_{}_image_{}.json"
CAPTION_PATH = "../comvg-caption-json/{}.json"

def read_image(id):
    return Image.open(IMAGE_PATH.format(id))

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

def get_matching(caption_id, image_id):
    f = open(CAPTION_PATH.format(caption_id, image_id))
    result = json.load(f)
    return result    

def black_outside_rectangle(image, left_top, right_bottom):
    # Create a new image with the same dimensions as the input image
    blacked_out_image = Image.new("RGB", image.size, color="black")

    # Create a mask with the rectangular region
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left_top, right_bottom], fill=255)

    # Paste the original image onto the blacked-out image using the mask
    blacked_out_image.paste(image, mask=mask)

    return blacked_out_image

def create_sub_image_obj(sentence_id, image_id):
    object_image = {}
    matched_objects = get_matching(sentence_id, image_id)
    image = Image.open(IMAGE_PATH.format(image_id))
    location = json.load(open(CAPTION_PATH.format(image_id)))
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

    # Create a mask to identify non-black pixels
    mask = Image.new("L", base_image.size, 0)

    # Iterate over the images and overlay non-black areas onto the black background
    for image in images:
        # Convert image to RGBA mode to handle transparency if needed
        image = image.convert("RGBA")

        # Create a mask for non-black pixels in the current image
        image_mask = image.convert("L")
        image_mask = image_mask.point(lambda x: 255 if x > 0 else 0, mode="1")

        # Paste non-black areas onto the black background
        base_image.paste(image, (0, 0), mask=image_mask)

    return base_image

def create_relation_object(object_images, subj, relation, obj, image_id, matched_json):
    relations = [relation]
    if type(relations) is not list or len(relations) == 0:
        return None, None
    verb_images = []
    verbs = []
    for relation in relations:
        if len(matched_json[subj]) == 0 and len(matched_json[obj]) != 0:
            verb_image = Image.open(IMAGE_PATH.format(image_id))
        elif len(matched_json[subj]) != 0 and len(matched_json[obj]) == 0:
            verb_image = Image.open(IMAGE_PATH.format(image_id))
        elif len(matched_json[subj]) != 0 and len(matched_json[obj]) != 0:
            verb_image = overlay_images([object_images[obj], object_images[subj]])
        else:
            verb_image = Image.open(IMAGE_PATH.format(image_id))
        verb_images.append(verb_image)
        verbs.append(relation)
    return verb_images, verbs

def normalize_tensor_list(tensor_list):
    total_sum = sum(tensor.item() for tensor in tensor_list)
    normalized_list = [tensor / total_sum for tensor in tensor_list]
    return normalized_list
