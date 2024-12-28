from torchvision import models
import torch.nn as nn
from torchvision.models import resnet18


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        resnet = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Use layers up to the penultimate
        self.feature_extractor = self.feature_extractor.to(device).eval()  # Move to device and set to eval
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        loss = nn.functional.l1_loss(input_features, target_features)
        return loss

