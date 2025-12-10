"""
Data loading and preprocessing utilities for 3D medical images
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from typing import Tuple, List


class MedicalImageDataset(Dataset):
    """
    3D Medical Image Dataset loader
    Handles NIFTI format images and masks
    """
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transforms=None):
        """
        Args:
            image_paths: List of paths to image volumes
            mask_paths: List of paths to segmentation masks
            transforms: Data augmentation transforms
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load NIFTI files
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()
        
        # Normalize image
        image = self._normalize(image)
        
        # Apply transforms if available
        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        # Convert to tensors
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        
        return image, mask
    
    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        mean = image.mean()
        std = image.std()
        if std == 0:
            return image
        return (image - mean) / (std + 1e-8)


def get_dataloaders(train_dir: str, val_dir: str, 
                   batch_size: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # This is a template - implement based on your data structure
    pass
