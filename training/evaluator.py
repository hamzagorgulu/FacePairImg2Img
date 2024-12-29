import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import numpy as np

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.lpips_metric = LPIPS(net='alex').to(device)  # Initialize LPIPS metric
    
    def calculate_ssim(self, image1, image2):
        """
        Calculate SSIM between two images.
        """
        # Convert tensors to numpy
        image1 = image1.permute(1, 2, 0).cpu().numpy()
        image2 = image2.permute(1, 2, 0).cpu().numpy()

        # Ensure images are in the range [0, 1]
        image1 = (image1 - image1.min()) / (image1.max() - image1.min())
        image2 = (image2 - image2.min()) / (image2.max() - image2.min())

        return ssim(image1, image2, multichannel=True, data_range=image1.max() - image1.min())
    
    def calculate_lpips(self, image1, image2):
        """
        Calculate LPIPS between two images.
        """
        image1 = image1.unsqueeze(0).to(self.device)
        image2 = image2.unsqueeze(0).to(self.device)

        return self.lpips_metric(image1, image2).item()
    
    def evaluate_batch(self, input_batch, target_batch, generated_batch):
        """
        Evaluate a batch of images with SSIM and LPIPS.
        """
        batch_size = input_batch.size(0)
        ssim_scores = []
        lpips_scores = []

        for i in range(batch_size):
            ssim_score = self.calculate_ssim(generated_batch[i], target_batch[i])
            lpips_score = self.calculate_lpips(generated_batch[i], target_batch[i])

            ssim_scores.append(ssim_score)
            lpips_scores.append(lpips_score)

        mean_ssim = np.mean(ssim_scores)
        mean_lpips = np.mean(lpips_scores)

        return mean_ssim, mean_lpips

    def evaluate(self, model, dataloader):
        """
        Evaluate the model on the validation dataset.
        """
        model.eval()
        total_ssim = 0
        total_lpips = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_images = batch['input'].to(self.device)
                target_images = batch['target'].to(self.device)

                generated_images = model(input_images)

                # Evaluate SSIM and LPIPS
                batch_ssim, batch_lpips = self.evaluate_batch(input_images, target_images, generated_images)
                total_ssim += batch_ssim
                total_lpips += batch_lpips
                num_batches += 1

        # Average SSIM and LPIPS over all batches
        avg_ssim = total_ssim / num_batches
        avg_lpips = total_lpips / num_batches

        return avg_ssim, avg_lpips
