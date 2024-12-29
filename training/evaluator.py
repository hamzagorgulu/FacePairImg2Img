import torch
import numpy as np
from torch.utils.data import DataLoader
import logging
from typing import Dict, List
from abc import ABC
from pytorch_fid import fid_score
import lpips
import kornia.losses as kl

class Evaluator(ABC):
    """Base evaluator class for paired image translation evaluation."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize metric modules."""
        self.ssim_module = kl.SSIM(window_size=11, reduction='mean')
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
        return float(1 - self.ssim_module(no_beard_images, beard_images))

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
        
        Args:
            test_dataset: Dataset containing paired beard/no-beard images
            
        Returns:
            dict: Dictionary containing all computed metrics
        """
        logging.info("Starting evaluation...")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,  # Adjust batch size as needed
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
                
                # Calculate per-batch metrics
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
