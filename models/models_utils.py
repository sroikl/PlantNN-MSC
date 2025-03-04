"""
Utility functions for model architectures.
"""
from typing import Tuple, Union, List, Optional

import torch
import torch.nn as nn


def greyscale_to_RGB(image: torch.Tensor, add_channels_dim: bool = False) -> torch.Tensor:
    """
    Convert a grayscale image to RGB by repeating the channel.
    
    Args:
        image: Input grayscale image tensor
        add_channels_dim: If True, adds a channel dimension before conversion
        
    Returns:
        RGB image tensor with 3 channels
    """
    if add_channels_dim:
        image = image.unsqueeze(-3)

    dims = [-1] * len(image.shape)
    dims[-3] = 3
    return image.expand(*dims)


class Identity(nn.Module):
    """
    Identity layer that returns the input unchanged.
    Used to replace layers in pretrained models when we want to extract features.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns input unchanged."""
        return x
