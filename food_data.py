import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import re
import  utils
from vlm_api import chat_image

categories = {
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad"
}

data_name = "food_train"
img_path = 'your_root_to_img'
ann_path = 'your_root_to_ann'
with open('your_root_to_text', 'r', encoding='utf-8') as f:
    image_list = [line.strip() for line in f.readlines()]
data = []
for i,img in tqdm(enumerate(image_list),total=len(image_list)):
    ai_ans = chat_image(os.path.join(img_path,img),categories)
    pred_ids = list(map(int, re.findall(r'\d+', ai_ans)))
    pred_ids = list(set(pred_ids))
    if pred_ids and max(pred_ids)>102:
        continue
    ann_img_path = os.path.join(ann_path, f"{img.split('.')[0]}.png")
    image = Image.open(ann_img_path).convert('L')
    pixel_values = np.array(image)
    gt_ids = np.unique(pixel_values)
    gt_ids = gt_ids[gt_ids != 0]
    gt_ids = gt_ids[gt_ids != 103]
    gt_ids = gt_ids.tolist()
    utils.json_write(data_name, img, ai_ans, pred_ids, gt_ids)
