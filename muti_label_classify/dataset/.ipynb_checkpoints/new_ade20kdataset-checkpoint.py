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

objs = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
    'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub',
    'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest',
    'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
    'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee', 'toilet',
    'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen',
    'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel',
    'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth',
    'television', 'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship',
    'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool',
    'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven',
    'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal',
    'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood',
    'sconce', 'vase', 'traffic', 'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate',
    'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag'
]
class Ade20kDetection(Dataset):
    def __init__(self,modify_json,ai_json,img_folder):
        self.modify_json = modify_json
        self.ai_json = ai_json
        self.img_folder = img_folder
        self.text_input = [f"a photo of {obj}" for obj in objs]
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
    def __init__(self, image_dir, modify_json_dir,ai_json_dir,input_transform=None):
        self.info = Ade20kDetection(modify_json=modify_json_dir,ai_json=ai_json_dir, img_folder=image_dir)
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
        self.input_transform = input_transform
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
            weight_01 = np.zeros_like(label_ai)
            weight_00 = 1 - weight_01
            weigth_idx = np.where(label_ai == 0)
            weigth_idx = np.array(weigth_idx).squeeze()
            input_text = []
            if weigth_idx.size:
                weigth_idx = np.atleast_1d(weigth_idx)
                for idx in weigth_idx:
                    input_text.append(f'this image contain {objs[idx]}')
                    input_text.append(f'this image not contain {objs[idx]}')
                similarity = self.clip.compute_similarity(image,input_text)
                top_indices = torch.topk(similarity[:, 0], k=min(3,similarity.shape[0])).indices
                softmax_similarity = torch.softmax(similarity[top_indices],dim=1).cpu().numpy()
                top_indices = top_indices.cpu().numpy()
                weight_01[top_indices] = softmax_similarity[:,0]
                weight_00[top_indices] = softmax_similarity[:,1]
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

    def getLabelVector(self, categories):
        label = np.zeros(150)
        # label_num = len(categories)
        for c in categories:
            label[c - 1] = 1.0  # / label_num
        return label

    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def __len__(self):
        return len(self.info)




