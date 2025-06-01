import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from PIL import Image


class CLIP(nn.Module):
    def __init__(self, num_classes,hidden_dim_1=1024,hidden_dim_2=256, clip_model_name='ViT-B/32'):
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


class CLIP_mld(nn.Module):
    def __init__(self, num_classes,hidden_dim_1=1024,hidden_dim_2=256, clip_model_name='ViT-B/32'):
        super(CLIP_mld, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.num_classes = num_classes
        self.num_features = hidden_dim_1
        self.relu = nn.ReLU()
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
        self.relu = nn.ReLU()
        self.num_features = fc_in_dim
        self.ml_decoder = None


    def forward(self, image_features):
            image_features = image_features.to(torch.float32)
            # x = image_features.unsqueeze(1)
            logits = self.ml_decoder(image_features)
            return logits

