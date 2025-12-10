"""
Metrics for medical image segmentation
"""

import torch
import numpy as np
from typing import Tuple


def dice_score(pred: torch.Tensor, target: torch.Tensor, 
               smooth: float = 1e-6) -> float:
    """
    Dice Score (F1 score for segmentation)
    
    Formula: Dice = 2|X∩Y| / (|X| + |Y|)
    
    Args:
        pred: Predicted segmentation (B, 1, D, H, W)
        target: Ground truth segmentation (B, 1, D, H, W)
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Dice score between 0 and 1
    """
    pred = (pred > 0.5).float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> float:
    """
    Intersection over Union (IoU) Score
    
    Formula: IoU = |X∩Y| / |X∪Y|
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        smooth: Smoothing constant
        
    Returns:
        IoU score between 0 and 1
    """
    pred = (pred > 0.5).float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def sensitivity_specificity(pred: torch.Tensor, target: torch.Tensor
                          ) -> Tuple[float, float]:
    """
    Sensitivity (Recall) and Specificity metrics
    Important for medical imaging (clinical relevance)
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        
    Returns:
        Tuple of (sensitivity, specificity)
    """
    pred = (pred > 0.5).float()
    
    # True positives, false negatives, true negatives, false positives
    tp = torch.sum(pred * target)
    fn = torch.sum((1 - pred) * target)
    tn = torch.sum((1 - pred) * (1 - target))
    fp = torch.sum(pred * (1 - target))
    
    sensitivity = tp / (tp + fn + 1e-6)  # True Positive Rate
    specificity = tn / (tn + fp + 1e-6)  # True Negative Rate
    
    return sensitivity.item(), specificity.item()


class SegmentationMetrics:
    """Container for segmentation metrics"""
    
    def __init__(self):
        self.dice_scores = []
        self.iou_scores = []
        self.sensitivities = []
        self.specificities = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update all metrics"""
        self.dice_scores.append(dice_score(pred, target))
        self.iou_scores.append(iou_score(pred, target))
        sens, spec = sensitivity_specificity(pred, target)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
    
    def get_summary(self) -> dict:
        """Get average metrics"""
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'sensitivity': np.mean(self.sensitivities),
            'specificity': np.mean(self.specificities)
        }
