from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity, FrechetInceptionDistance
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Union
from abc import ABC, abstractmethod

class Evaluator(ABC):
    """Base evaluator class that implements common image translation metrics."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize evaluator with device specification.
        
        Args:
            device: Device to run evaluations on ('cuda' or 'cpu')
        """
        self.device = device
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize all metric modules."""
        self.ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_module = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)
        self.fid_module = FrechetInceptionDistance(normalize=True).to(self.device)
        
    def calculate_ssim(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate SSIM between real and generated images.
        
        Intuition: SSIM measures structural similarity between images. For beard removal,
        we expect moderate SSIM scores since we want to preserve overall facial structure
        while changing the beard region. Very high SSIM might indicate insufficient beard
        removal.
        
        Args:
            real_images: Tensor of real images [B, C, H, W]
            generated_images: Tensor of generated images [B, C, H, W]
        
        Returns:
            float: Mean SSIM score
        """
        return float(self.ssim_module(generated_images, real_images))

    def calculate_lpips(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Calculate LPIPS between real and generated images.
        
        Intuition: LPIPS captures perceptual similarity using deep features. For beard removal,
        it helps ensure the generated faces look natural and maintain identity, while allowing
        for the desired beard removal changes.
        
        Args:
            real_images: Tensor of real images [B, C, H, W]
            generated_images: Tensor of generated images [B, C, H, W]
        
        Returns:
            float: Mean LPIPS score
        """
        return float(self.lpips_module(generated_images, real_images).mean())

    def calculate_fid(self, real_loader: DataLoader, generated_loader: DataLoader) -> float:
        """
        Calculate FID between real and generated image distributions.
        
        Intuition: FID measures both quality and diversity of generated images. For beard removal,
        it helps ensure generated clean-shaven faces maintain realism and natural variation,
        rather than converging to average faces.
        
        Args:
            real_loader: DataLoader for real images
            generated_loader: DataLoader for generated images
        
        Returns:
            float: FID score
        """
        self.fid_module.reset()
        
        # Process real images
        for batch in real_loader:
            images = batch[0].to(self.device)
            self.fid_module.update(images, real=True)
            
        # Process generated images
        for batch in generated_loader:
            images = batch[0].to(self.device)
            self.fid_module.update(images, real=False)
            
        return float(self.fid_module.compute())

    @abstractmethod
    def get_generated_images(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to generate images from input batch.
        Must be implemented by child classes.
        """
        pass

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Run comprehensive evaluation using all metrics.
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            dict: Dictionary containing all computed metrics
        """
        logging.info("Starting evaluation...")
        
        real_batches: List[torch.Tensor] = []
        generated_batches: List[torch.Tensor] = []
        ssim_scores: List[float] = []
        lpips_scores: List[float] = []
        
        with torch.no_grad():
            for batch in test_loader:
                real_images = batch[0].to(self.device)
                generated_images = self.get_generated_images(real_images)
                
                # Calculate per-batch metrics
                ssim_scores.append(self.calculate_ssim(real_images, generated_images))
                lpips_scores.append(self.calculate_lpips(real_images, generated_images))
                
                # Store for FID calculation
                real_batches.append(real_images.cpu())
                generated_batches.append(generated_images.cpu())
        
        # Create separate dataloaders for FID
        real_fid_loader = DataLoader(
            torch.cat(real_batches, 0),
            batch_size=test_loader.batch_size
        )
        generated_fid_loader = DataLoader(
            torch.cat(generated_batches, 0),
            batch_size=test_loader.batch_size
        )
        
        # Calculate FID
        fid_score = self.calculate_fid(real_fid_loader, generated_fid_loader)
        
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