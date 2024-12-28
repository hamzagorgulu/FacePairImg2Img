import torch
from torchvision.utils import save_image
import os

class BeardRemovalTester:
    def __init__(self, model, test_dataset, device):
        self.model = model.to(device)
        self.device = device
        self.test_dataset = test_dataset
        
        os.makedirs('test_results', exist_ok=True)
    
    def test(self, num_samples=5):
        self.model.eval()
        
        with torch.no_grad():
            for i in range(num_samples):
                sample = self.test_dataset[i]
                input_image = sample['input'].unsqueeze(0).to(self.device)
                target_image = sample['target']
                
                generated_image = self.model(input_image)

                input_image = input_image.cpu()
                target_image = target_image.cpu()
                generated_image = generated_image.cpu()
                
                # Save results
                save_image(
                    [input_image.squeeze(0), generated_image.squeeze(0), target_image],
                    f'test_results/test_sample_{i}.png',
                    normalize=True
                )