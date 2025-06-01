import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from models import clip_muti
import numpy as np
import json
import random
from tqdm import tqdm


category_map = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
    18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
    35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
    50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
    64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
    82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
}



class CocoDetection(Dataset):
    def __init__(self,modify_json,ai_json,img_folder):
        self.modify_json = modify_json
        self.ai_json = ai_json
        self.img_folder = img_folder
        with open(self.modify_json, 'r') as f:
            modify_js = json.load(f)
        self.coco_modify = [info for info in modify_js]
        with open(self.ai_json, 'r') as f:
            ai_js = json.load(f)
        self.coco_ai = [info for info in ai_js]
        self.img_names = [list(info.keys())[0] for info in modify_js]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        target_modify = self.coco_modify[idx][img_name]
        target_ai = self.coco_ai[idx][img_name]
        return image, target_modify,target_ai

class CoCoDataset(data.Dataset):
    def __init__(self, image_dir, modify_json_dir,ai_json_dir,input_transform=None,type=None):
        self.coco = CocoDetection(modify_json=modify_json_dir,ai_json=ai_json_dir, img_folder=image_dir)
        self.category_map = category_map
        self.input_transform = input_transform
        self.pre_weight = []
        self.img_names = []
        with open('your_root_to_tagclip_txt', 'r') as f:
            for line in f:
                parts = line.strip().split()
                self.img_names.append(parts[0])
                self.pre_weight.append([float(x) for x in parts[1:]])
        self.pre_weight = np.array(self.pre_weight, dtype=np.float32)
        self.clip = clip_muti.CLIP(len(objs))
        self.modify_labels = []
        self.ai_labels = []
        self.weight_00 = []
        self.weight_01 = []
        self.images = []
        l = len(self.coco)
        for i in tqdm(range(l)):
            image, target_modify, target_ai = self.coco[i]
            image = self.clip.encode_image(image)
            label_modify = self.getLabelVector(target_modify)
            label_ai = self.getLabelVector(target_ai)
            if type == 'val':
                weight_01 = 0
                weight_00 = 1 - weight_01
            else:
                weight_01 = self.pre_weight[i]
                weight_00 = 1 - weight_01
            self.modify_labels.append(label_modify)
            self.ai_labels.append(label_ai)
            self.weight_00.append(weight_00)
            self.weight_01.append(weight_01)
            self.images.append(image)
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        input = self.images[index]
        if self.input_transform:
            input = self.input_transform(input)

        return input, self.modify_labels[index],self.weight_00[index],self.weight_01[index],self.ai_labels[index]

    def update(self,new_weight_01,new_weight_00):
        self.weight_01 = new_weight_01
        self.weight_00 = new_weight_00
    def getLabelVector(self, categories):
        label = np.zeros(80)
        # label_num = len(categories)
        for c in categories:
            if c not in self.category_map.keys():
                continue
            index = self.category_map[c]-1
            label[index] = 1.0 # / label_num
        return label

    def __len__(self):
        return len(self.coco)






