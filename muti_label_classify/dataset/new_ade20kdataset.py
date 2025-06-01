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

class Ade20kDetection(Dataset):
    def __init__(self,modify_json,ai_json,img_folder):
        self.modify_json = modify_json
        self.ai_json = ai_json
        self.img_folder = img_folder
        with open(self.modify_json, 'r') as f:
            modify_js = json.load(f)
        self.ade20k_modify = [info for info in modify_js]
        with open(self.ai_json, 'r') as f:
            ai_js = json.load(f)
        self.ade20k_ai = [info for info in ai_js]
        self.img_names = [list(info.keys())[0] for info in modify_js]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        target_modify = self.ade20k_modify[idx][img_name]
        target_ai = self.ade20k_ai[idx][img_name]
        return image, target_modify,target_ai

class Ade20kDataset(data.Dataset):
    def __init__(self, image_dir, modify_json_dir,ai_json_dir,input_transform=None,type=None):
        self.info = Ade20kDetection(modify_json=modify_json_dir,ai_json=ai_json_dir, img_folder=image_dir)
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
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
        l = len(self.info)
        for i in tqdm(range(l)):
            image, target_modify, target_ai = self.info[i]
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

    def update(self, new_weight_01, new_weight_00):
        self.weight_01 = torch.tensor(new_weight_01, dtype=torch.float32)
        self.weight_00 = torch.tensor(new_weight_00, dtype=torch.float32)

    def getLabelVector(self, categories):
        label = np.zeros(150)
        # label_num = len(categories)
        for c in categories:
            label[c - 1] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.info)




