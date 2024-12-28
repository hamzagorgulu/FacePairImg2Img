from torchvision import models
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3']):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.selected_layers = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15
        }
        self.model = nn.Sequential(*list(vgg.children())[:16]).eval()  # Up to relu3_3
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features, target_features = {}, {}
        for name, layer in self.selected_layers.items():
            input_features[name] = self.model[:layer](input)
            target_features[name] = self.model[:layer](target)
        loss = sum([nn.functional.l1_loss(input_features[name], target_features[name]) for name in self.layers])
        return loss
