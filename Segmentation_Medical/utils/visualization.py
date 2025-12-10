"""
Visualization utilities for 3D medical images
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def visualize_3d_volume(volume: np.ndarray, title: str = "3D Volume"):
    """
    Interactive 3D volume visualization with slider
    
    Args:
        volume: 3D numpy array (D, H, W)
        title: Title for the figure
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    # Display middle slice initially
    slice_idx = volume.shape[0] // 2
    im = ax.imshow(volume[slice_idx], cmap='gray')
    ax.set_title(f"{title} (Slice {slice_idx})")
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0]-1, 
                    valinit=slice_idx, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        im.set_data(volume[idx])
        ax.set_title(f"{title} (Slice {idx})")
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


def compare_segmentations(image: np.ndarray, pred: np.ndarray, 
                         target: np.ndarray, slice_idx: int = None):
    """
    Compare prediction and ground truth segmentation
    
    Args:
        image: Original image volume (D, H, W)
        pred: Predicted segmentation (D, H, W)
        target: Ground truth segmentation (D, H, W)
        slice_idx: Slice index to display (default: middle slice)
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    # Original image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(target[slice_idx], cmap='Reds', alpha=0.7)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred[slice_idx], cmap='Blues', alpha=0.7)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay: Green=TP, Red=FP, Yellow=FN
    overlay = np.zeros((*image[slice_idx].shape, 3))
    overlay[..., 0] = (pred[slice_idx] * (1 - target[slice_idx]))  # False Positives
    overlay[..., 1] = ((pred[slice_idx] * target[slice_idx]) / 2)   # True Positives
    overlay[..., 2] = ((1 - pred[slice_idx]) * target[slice_idx])   # False Negatives
    
    axes[3].imshow(image[slice_idx], cmap='gray')
    axes[3].imshow(overlay, alpha=0.5)
    axes[3].set_title('TP (Green) / FP (Red) / FN (Blue)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
