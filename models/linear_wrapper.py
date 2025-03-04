"""
Model wrapper for adding linear classification layers.
"""
from typing import Any, Tuple, Dict, List, Optional, Union

import torch
import torch.nn as nn


class LinearWrapper(nn.Module):
    """
    Wraps a feature extractor module with a linear layer for classification.
    
    This wrapper adds a fully connected layer after a feature extractor
    to convert embeddings into class probabilities.
    """

    def __init__(
        self, 
        extractor: nn.Module, 
        embedding_dim: int, 
        num_classes: int, 
        bias: bool = True
    ):
        """
        Initialize the linear wrapper.
        
        Args:
            extractor: Feature extractor module to wrap
            embedding_dim: Dimension of the feature extractor's output
            num_classes: Number of output classes
            bias: Whether to include bias in the linear layer
        """
        super().__init__()

        self.module = extractor
        self.fc = nn.Linear(embedding_dim, num_classes, bias=bias)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Extract features and classify them.
        
        Args:
            *args: Positional arguments to pass to the feature extractor
            **kwargs: Keyword arguments to pass to the feature extractor
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.module(*args, **kwargs)
        
        # Classify features
        logits = self.fc(features)
        
        return logits
    
    def get_embedding(self, *args, **kwargs) -> torch.Tensor:
        """
        Get embeddings without classification.
        
        Args:
            *args: Positional arguments to pass to the feature extractor
            **kwargs: Keyword arguments to pass to the feature extractor
            
        Returns:
            Feature embeddings
        """
        return self.module(*args, **kwargs)
