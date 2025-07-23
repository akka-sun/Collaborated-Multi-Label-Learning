from pycocotools.coco import COCO
import numpy as np
from vlm_api import chat_image
from tqdm import tqdm
import os
import re
import utils

categories = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

data_name = "coco_train"
img_path = "your_root_to_data"
coco=COCO('your_root_to_coco_json')
for i,img in tqdm(enumerate(os.listdir(img_path)),total=len(os.listdir(img_path))):
    ai_ans = chat_image(os.path.join(img_path,img),categories)
    pred_ids = list(map(int, re.findall(r'\d+', ai_ans)))
    pred_ids = set(pred_ids)
    if pred_ids and max(pred_ids)>90:
        continue
    id, _ = os.path.splitext(img)
    id = int(id.lstrip('0'))
    img_info = coco.loadImgs([id])
    img_id = img_info[0]['id']
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    gt_ids = [ann['category_id'] for ann in anns]
    gt_ids = sorted(set(gt_ids))
    utils.json_write(data_name,img,ai_ans,pred_ids,gt_ids)
