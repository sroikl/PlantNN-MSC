"""
Feature extraction models for plant classification.

This module contains various models for extracting features from plant images
across different modalities (LWIR, VIR, etc.).
"""
from typing import Dict, Union, List, Tuple, Any, Optional, TypeVar

import torch
import torch.nn as nn
from torchvision.models import inception_v3
from TCN.tcn import TemporalConvNet

from .utils import Identity, greyscale_to_RGB


class ImageFeatureExtractor(nn.Module):
    """
    Extract features from images using a pre-trained InceptionV3 model.
    
    This model uses a frozen InceptionV3 network with the classification layer
    replaced by an identity function to extract features from images.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the image feature extractor.
        
        Args:
            pretrained: Whether to use pre-trained weights
        """
        super().__init__()

        self.inception = inception_v3(
            pretrained=pretrained, 
            transform_input=False, 
            aux_logits=True
        )
        self.inception.fc = Identity()
        self.inception.eval()

        # Freeze all parameters
        for p in self.inception.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "ImageFeatureExtractor":
        """
        Override train method to ensure the model stays in eval mode.
        
        Args:
            mode: Training mode flag (ignored)
            
        Returns:
            Self instance
        """
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of images.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, channels, height, width)
               or (batch_size, seq_len, height, width) for grayscale
               
        Returns:
            Feature tensor of shape (batch_size, seq_len, 2048)
        """
        # Convert grayscale to RGB if needed
        if len(x.shape) < 5 or (len(x.shape) >= 5 and x.shape[-3] == 1):
            x = greyscale_to_RGB(x, add_channels_dim=(len(x.shape) < 5))

        # Process each image in the batch sequence
        N, T = x.shape[:2]
        return self.inception(x.view(-1, *x.shape[2:])).view(N, T, -1)


class ModalityFeatureExtractor(TemporalConvNet):
    """
    Extract temporal features from image sequence features using TCN.
    
    This model applies temporal convolutions to process image feature sequences.
    """

    def __init__(
        self, 
        num_levels: int = 3, 
        num_hidden: int = 600, 
        embedding_size: int = 128, 
        kernel_size: int = 2,
        dropout: float = 0.2,
        input_size: int = 2048
    ):
        """
        Initialize the modality feature extractor.
        
        Args:
            num_levels: Number of TCN layers
            num_hidden: Number of channels in hidden layers
            embedding_size: Size of final feature vector
            kernel_size: Kernel size for temporal convolutions
            dropout: Dropout probability
            input_size: Input feature dimension (from ImageFeatureExtractor)
        """
        num_channels = [num_hidden] * (num_levels - 1) + [embedding_size]
        super().__init__(input_size, num_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features from a sequence of image features.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               where input_size is typically 2048 from InceptionV3
               
        Returns:
            Feature tensor of shape (batch_size, seq_len, embedding_size)
        """
        # Transpose for TCN processing (expects [N, C, L])
        x = torch.stack([m.t() for m in x])
        
        # Process through temporal convolution
        out = super().forward(x)
        
        # Transpose back to original format
        return torch.stack([m.t() for m in out])


class PlantFeatureExtractor(nn.Module):
    """
    Multi-modal feature extractor for plant classification.
    
    This model extracts features from multiple modalities (LWIR, VIR, etc.)
    and combines them into a single feature vector.
    """

    def __init__(
        self, 
        *default_mods: str, 
        embedding_size: int = 512, 
        modality_embedding_size: int = 128,
        **param_mods: Dict[str, Union[int, float]]
    ):
        """
        Initialize the plant feature extractor.
        
        Args:
            *default_mods: Modality names to use with default parameters
            embedding_size: Size of the final feature vector
            modality_embedding_size: Size of each modality's embedding
            **param_mods: Modality names with custom parameters
        """
        super().__init__()
        
        # Validate modality configuration
        assert len(set(default_mods).intersection(param_mods.keys())) == 0, \
            "A modality cannot be both default and parameterized"

        # Store all modalities
        self.mods = list(default_mods) + list(param_mods.keys())
        self.modality_embedding_size = modality_embedding_size
        
        # Validate we have at least one modality
        assert len(self.mods) > 0, "At least one modality must be specified"

        # Create image feature extractor (shared across modalities)
        self.image_feat_ext = ImageFeatureExtractor()

        # Create modality-specific feature extractors
        self.mod_extractors: Dict[str, nn.Module] = {}
        
        # Initialize default modality extractors
        for mod in default_mods:
            self.mod_extractors[mod] = ModalityFeatureExtractor(
                embedding_size=modality_embedding_size
            )
            self.add_module(f'TCN_{mod}_feat_extractor', self.mod_extractors[mod])

        # Initialize parameterized modality extractors
        for mod, params in param_mods.items():
            self.mod_extractors[mod] = ModalityFeatureExtractor(
                embedding_size=modality_embedding_size,
                **params
            )
            self.add_module(f'TCN_{mod}_feat_extractor', self.mod_extractors[mod])

        # Final feature fusion layer
        self.final_feat_extractor = nn.Linear(
            modality_embedding_size * len(self.mods), 
            embedding_size
        )

        # CUDA stream setup
        self.device = None
        self.streams = {mod: None for mod in self.mods}

    def to(self, *args, **kwargs) -> "PlantFeatureExtractor":
        """
        Move model to device and set up CUDA streams if needed.
        
        Args:
            *args: Arguments to pass to the parent to() method
            **kwargs: Keyword arguments to pass to the parent to() method
            
        Returns:
            Self instance
        """
        super().to(*args, **kwargs)

        # Set up CUDA streams if we're on a GPU
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).get_device()
            if self.device != device:
                self.device = device
                self.streams = {mod: torch.cuda.Stream(device=self.device) for mod in self.mods}
        elif self.device is not None:
            self.device = None
            self.streams = {mod: None for mod in self.mods}

        return self

    def forward(self, **x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from multiple modalities and fuse them.
        
        Args:
            **x: Input tensors for each modality, where each tensor has shape
                (batch_size, seq_len, channels, height, width)
                
        Returns:
            Fused feature vector of shape (batch_size, embedding_size)
        """
        # Validate that all modalities are provided
        assert set(self.mods) == set(x.keys()), \
            f"Expected modalities {self.mods} but got {list(x.keys())}"

        # Extract image features (with parallelism if on CUDA)
        if self.device is None:
            # CPU processing (sequential)
            img_feats = {mod: self.image_feat_ext(x[mod]) for mod in self.mods}
        else:
            # GPU processing (parallel streams)
            img_feats = {}
            for mod in self.mods:
                with torch.cuda.stream(self.streams[mod]):
                    img_feats[mod] = self.image_feat_ext(x[mod])

        # Extract modality-specific features
        if self.device is None:
            # CPU processing (sequential)
            mod_feats = {mod: self.mod_extractors[mod](img_feats[mod]) for mod in self.mods}
        else:
            # GPU processing (parallel streams)
            mod_feats = {}
            for mod in self.mods:
                with torch.cuda.stream(self.streams[mod]):
                    mod_feats[mod] = self.mod_extractors[mod](img_feats[mod])
            # Synchronize streams
            for mod in self.mods:
                self.streams[mod].synchronize()

        # Extract final frame features from each sequence
        combined_features = torch.cat([mod_feats[mod][:, -1, :] for mod in self.mods], dim=1)

        # Final feature fusion
        return self.final_feat_extractor(combined_features)
