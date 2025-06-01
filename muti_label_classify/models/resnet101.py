import torch
import torch.nn as nn
import torchvision.models as models

class ResNet101_Multiclass(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet101_Multiclass, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet101(x)
        return x