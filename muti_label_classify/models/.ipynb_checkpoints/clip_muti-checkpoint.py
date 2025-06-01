import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from PIL import Image


class CLIP(nn.Module):
    def __init__(self, num_classes,hidden_dim_1=1024,hidden_dim_2=256, clip_model_name='ViT-B/32'):
        """
        :param num_classes: 输出类别数量。
        :param clip_model_name: 使用的 CLIP 模型名称，默认为 'ViT-B/32'。
        """
        super(CLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        if hasattr(self.clip_model.visual, 'proj'):
            fc_in_dim = self.clip_model.visual.proj.shape[1]
        elif hasattr(self.clip_model.visual, 'output_dim'):
            fc_in_dim = self.clip_model.visual.output_dim
        else:
            fc_in_dim = 512
        self.fc1 = nn.Linear(fc_in_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, num_classes)
        self.relu = nn.ReLU()


    def forward(self, image_features):
            """
            :param images: 预处理后的图像 Tensor，形状为 (B, C, H, W)。
            :return: Tensor，输出 logits，形状为 (B, num_classes)。
            """
            image_features = image_features.to(torch.float32)
            x = self.relu(self.fc1(image_features))
            x = self.relu(self.fc2(x))
            logits = self.fc3(x)
            return logits

    def encode_text(self, text):
        with torch.no_grad():
            text_tokens = clip.tokenize(text).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
        return text_features

    def encode_image(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        return image_features

    def compute_similarity(self, image_features, text):
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T)
        similarity = similarity.squeeze().view(-1,2)
        # similarity_1 = similarity[:,:3].mean(dim=1,keepdim=True)
        # similarity_2 = similarity[:,3:].mean(dim=1,keepdim=True)
        # similarity = torch.cat((similarity_1, similarity_2), dim=1)
        # softmax_similarity = F.softmax(similarity, dim=1).tolist()
        return similarity


class CLIP_change(nn.Module):
    def __init__(self, num_classes,hidden_dim_1=2048,hidden_dim_2=4096,hidden_dim_3=512, clip_model_name='ViT-B/32'):
        """
        :param num_classes: 输出类别数量。
        :param clip_model_name: 使用的 CLIP 模型名称，默认为 'ViT-B/32'。
        """
        super(CLIP_change, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        if hasattr(self.clip_model.visual, 'proj'):
            fc_in_dim = self.clip_model.visual.proj.shape[1]
        elif hasattr(self.clip_model.visual, 'output_dim'):
            fc_in_dim = self.clip_model.visual.output_dim
        else:
            fc_in_dim = 512
        self.fc1 = nn.Linear(fc_in_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3,num_classes)
        self.relu = nn.ReLU()


    def forward(self, image_features):
            """
            :param images: 预处理后的图像 Tensor，形状为 (B, C, H, W)。
            :return: Tensor，输出 logits，形状为 (B, num_classes)。
            """
            image_features = image_features.to(torch.float32)
            x = self.relu(self.fc1(image_features))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            logits = self.fc4(x)
            return logits




if __name__ =='__main__':
    objs = [
    'person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
    clip_sim = CLIP(80)
    image_path = "/root/autodl-tmp/train2017/000000281563.jpg"  # 替换为你的图像路径
    image = Image.open(image_path).convert("RGB")
    text_input_1 = []
    for obj in objs:
        text_input_1.append(f'a image with {obj}')
        text_input_1.append(f'a image without {obj}')
    image = clip_sim.encode_image(image)
    sim_score_1 = clip_sim.compute_similarity(image, text_input_1)
    text_input_2=[]
    for obj in objs:
        text_input_2.append(f'a image with {obj}')
    text_features = clip_sim.encode_text(text_input_2)
    similarity = (image @ text_features.T)
    similarity = similarity.squeeze()
    sim_score_2 = F.softmax(similarity, dim=0).tolist()
    for i,obj in enumerate(objs):
        print(f'{obj}:{sim_score_1[i]}----{sim_score_2[i]}-{1-sim_score_2[i]}')

