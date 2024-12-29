import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import logging
from typing import Dict, List
from abc import ABC
from pytorch_fid import fid_score
import lpips
import kornia.losses as kl

def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Create a Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    
    g = coords.pow(2)
    g = (-g / (2 * sigma ** 2)).exp()
    
    return g / g.sum()

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Create a SSIM window."""
    _1D_window = gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Calculate SSIM between two images.
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of the sliding window
        
    Returns:
        torch.Tensor: SSIM score
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Create window
    window = create_window(window_size, img1.size(1)).to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

class Evaluator(ABC):
    """Base evaluator class for paired image translation evaluation."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize metric modules."""
        self.lpips_module = lpips.LPIPS(net='alex').to(self.device)
        
    def calculate_ssim(self, beard_images: torch.Tensor, no_beard_images: torch.Tensor) -> float:
        """
        Calculate SSIM between beard and no-beard images.
        
        Intuition: SSIM measures structural similarity. For beard removal evaluation,
        moderate SSIM scores are expected as facial structure should be preserved
        while beard region changes.
        
        Args:
            beard_images: Tensor of images with beards [B, C, H, W]
            no_beard_images: Tensor of images without beards [B, C, H, W]
        
        Returns:
            float: Mean SSIM score (higher is better)
        """
        return float(calculate_ssim(no_beard_images, beard_images))

    def calculate_lpips(self, beard_images: torch.Tensor, no_beard_images: torch.Tensor) -> float:
        """
        Calculate LPIPS between beard and no-beard images.
        
        Intuition: LPIPS captures perceptual similarity using deep features. For beard removal,
        it helps evaluate if the no-beard images maintain identity while allowing for
        the desired beard removal changes.
        
        Args:
            beard_images: Tensor of images with beards [B, C, H, W]
            no_beard_images: Tensor of images without beards [B, C, H, W]
        
        Returns:
            float: Mean LPIPS score (lower is better)
        """
        with torch.no_grad():
            return float(self.lpips_module(no_beard_images, beard_images).mean())

    def calculate_fid(self, beard_paths: List[str], no_beard_paths: List[str]) -> float:
        """
        Calculate FID between beard and no-beard image distributions.
        
        Intuition: FID measures quality and diversity of the no-beard distribution
        compared to the beard distribution. It helps ensure the no-beard images
        maintain natural variation and don't converge to average faces.
        
        Args:
            beard_paths: List of paths to beard images
            no_beard_paths: List of paths to no-beard images
        
        Returns:
            float: FID score (lower is better)
        """
        return float(fid_score.calculate_fid_given_paths([beard_paths, no_beard_paths]))

    def evaluate(self, test_dataset):
        """
        Run comprehensive evaluation using all metrics.
        """
        logging.info("Starting evaluation...")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False
        )
        
        ssim_scores: List[float] = []
        lpips_scores: List[float] = []
        beard_paths: List[str] = []
        no_beard_paths: List[str] = []
        
        with torch.no_grad():
            for beard_images, no_beard_images in test_loader:
                beard_images = beard_images.to(self.device)
                no_beard_images = no_beard_images.to(self.device)
                
                ssim_scores.append(self.calculate_ssim(beard_images, no_beard_images))
                lpips_scores.append(self.calculate_lpips(beard_images, no_beard_images))
                
            # Collect paths for FID calculation
            for pair in test_dataset.image_pairs:
                beard_paths.append(pair[1])
                no_beard_paths.append(pair[2])
        
        # Calculate FID
        fid_score = self.calculate_fid(beard_paths, no_beard_paths)
        
        # Calculate mean scores
        results = {
            'ssim': float(np.mean(ssim_scores)),
            'lpips': float(np.mean(lpips_scores)),
            'fid': fid_score
        }
        
        # Log results
        logging.info("Evaluation Results:")
        for metric, value in results.items():
            logging.info(f"{metric.upper()} Score: {value:.4f}")
            
        return results