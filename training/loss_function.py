from torchvision import models
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, device="cuda", layers=['relu1_2', 'relu2_2', 'relu3_3']):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers
        self.selected_layers = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15
        }
        self.model = nn.Sequential(*list(vgg.children())[:16]).to(device).eval()  # Move VGG to device
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        self.device = device

    def forward(self, input, target):
        input = input.to(self.device)  # Ensure input is on the same device
        target = target.to(self.device)  # Ensure target is on the same device
        
        input_features, target_features = {}, {}
        for name, layer in self.selected_layers.items():
            input_features[name] = self.model[:layer](input)
            target_features[name] = self.model[:layer](target)
        loss = sum([nn.functional.l1_loss(input_features[name], target_features[name]) for name in self.layers])
        return loss

