"""
Custom transformations for image data preprocessing.
"""
from PIL import Image
import random
from typing import Tuple, Union, Any, Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class BaseRandomTransform:
    """
    Base class for random transformations that maintain consistency across sequences.
    
    This ensures the same random parameter is applied to all images in a sequence.
    """
    
    def __init__(self, sequence_length: int):
        """
        Initialize the transform.
        
        Args:
            sequence_length: Length of the image sequence
        """
        self.sequence_length = sequence_length
        self.current_position = 0
        self.current_params = None
        
    def get_params(self) -> Any:
        """
        Generate random parameters for the transformation.
        
        Returns:
            Parameters for the transformation
        """
        raise NotImplementedError("Subclasses must implement get_params")
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], params: Any) -> Union[Image.Image, torch.Tensor]:
        """
        Apply the transformation with the given parameters.
        
        Args:
            img: Input image
            params: Parameters for the transformation
            
        Returns:
            Transformed image
        """
        raise NotImplementedError("Subclasses must implement apply_transform")
        
    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """
        Apply the transformation to an image.
        
        Args:
            img: Input image
            
        Returns:
            Transformed image
        """
        # Generate new parameters at the start of each sequence
        if self.current_position == 0:
            self.current_params = self.get_params()
            
        # Apply transformation with current parameters
        result = self.apply_transform(img, self.current_params)
        
        # Update position in sequence
        self.current_position = (self.current_position + 1) % self.sequence_length
        
        return result


class RandomBrightness(BaseRandomTransform):
    """Apply random brightness adjustment to a sequence of images."""
    
    def get_params(self) -> float:
        """
        Generate random brightness factor.
        
        Returns:
            Brightness factor between 0.5 and 2.0
        """
        return random.uniform(0.5, 2.0)
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], brightness_factor: float) -> Union[Image.Image, torch.Tensor]:
        """
        Apply brightness adjustment.
        
        Args:
            img: Input image
            brightness_factor: Brightness adjustment factor
            
        Returns:
            Brightness-adjusted image
        """
        return TF.adjust_brightness(img, brightness_factor)


class RandomContrast(BaseRandomTransform):
    """Apply random contrast adjustment to a sequence of images."""
    
    def get_params(self) -> float:
        """
        Generate random contrast factor.
        
        Returns:
            Contrast factor between 0.5 and 2.0
        """
        return random.uniform(0.5, 2.0)
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], contrast_factor: float) -> Union[Image.Image, torch.Tensor]:
        """
        Apply contrast adjustment.
        
        Args:
            img: Input image
            contrast_factor: Contrast adjustment factor
            
        Returns:
            Contrast-adjusted image
        """
        return TF.adjust_contrast(img, contrast_factor)


class RandomGamma(BaseRandomTransform):
    """Apply random gamma adjustment to a sequence of images."""
    
    def get_params(self) -> Tuple[float, float]:
        """
        Generate random gamma and gain factors.
        
        Returns:
            Tuple of (gamma, gain) factors
        """
        gamma = random.uniform(0.5, 2.0)
        gain = random.uniform(0.5, 2.0)
        return gamma, gain
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], params: Tuple[float, float]) -> Union[Image.Image, torch.Tensor]:
        """
        Apply gamma adjustment.
        
        Args:
            img: Input image
            params: Tuple of (gamma, gain) factors
            
        Returns:
            Gamma-adjusted image
        """
        gamma, gain = params
        return TF.adjust_gamma(img, gamma, gain)


class RandomHue(BaseRandomTransform):
    """Apply random hue adjustment to a sequence of images."""
    
    def get_params(self) -> float:
        """
        Generate random hue factor.
        
        Returns:
            Hue adjustment factor
        """
        return random.uniform(0.5, 2.0)
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], hue_factor: float) -> Union[Image.Image, torch.Tensor]:
        """
        Apply hue adjustment.
        
        Args:
            img: Input image
            hue_factor: Hue adjustment factor
            
        Returns:
            Hue-adjusted image
        """
        return TF.adjust_hue(img, hue_factor)


class RandomSaturation(BaseRandomTransform):
    """Apply random saturation adjustment to a sequence of images."""
    
    def get_params(self) -> float:
        """
        Generate random saturation factor.
        
        Returns:
            Saturation adjustment factor
        """
        return random.uniform(0.5, 2.0)
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], saturation_factor: float) -> Union[Image.Image, torch.Tensor]:
        """
        Apply saturation adjustment.
        
        Args:
            img: Input image
            saturation_factor: Saturation adjustment factor
            
        Returns:
            Saturation-adjusted image
        """
        return TF.adjust_saturation(img, saturation_factor)


class RandomCrop(BaseRandomTransform):
    """Apply random crop to a sequence of images."""
    
    def __init__(self, sequence_length: int, output_size: Union[int, Tuple[int, int]]):
        """
        Initialize random crop transform.
        
        Args:
            sequence_length: Length of the image sequence
            output_size: Target output size (as tuple or single int)
        """
        super().__init__(sequence_length)
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
            
    def get_params(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Generate random crop parameters.
        
        Returns:
            Tuple of (top, left, height, width) or None if image is too small
        """
        # Note: This relies on the first image in the sequence to determine crop parameters
        # Actual parameters will be set during the first __call__
        return None
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], params: Tuple[int, int, int, int]) -> Union[Image.Image, torch.Tensor]:
        """
        Apply crop with given parameters.
        
        Args:
            img: Input image
            params: Tuple of (top, left, height, width)
            
        Returns:
            Cropped image
        """
        # If parameters are None, compute them now based on this image
        if params is None:
            # For PIL images
            if isinstance(img, Image.Image):
                width, height = img.size
            # For tensors
            else:
                if img.dim() == 3:  # C, H, W
                    height, width = img.shape[1], img.shape[2]
                elif img.dim() == 2:  # H, W
                    height, width = img.shape
                    
            if height < self.output_size[0] or width < self.output_size[1]:
                return TF.center_crop(img, min(height, width))
                
            # Compute crop parameters
            top = random.randint(0, height - self.output_size[0])
            left = random.randint(0, width - self.output_size[1])
            
            self.current_params = (top, left, self.output_size[0], self.output_size[1])
            
        return TF.crop(img, *self.current_params)


class RandomHorizontalFlip(BaseRandomTransform):
    """Apply random horizontal flip to a sequence of images."""
    
    def __init__(self, sequence_length: int, p: float = 0.5):
        """
        Initialize random horizontal flip transform.
        
        Args:
            sequence_length: Length of the image sequence
            p: Probability of applying the flip
        """
        super().__init__(sequence_length)
        self.p = p
        
    def get_params(self) -> bool:
        """
        Determine whether to apply flip.
        
        Returns:
            True if flip should be applied, False otherwise
        """
        return random.random() < self.p
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], flip: bool) -> Union[Image.Image, torch.Tensor]:
        """
        Apply horizontal flip if indicated.
        
        Args:
            img: Input image
            flip: Whether to flip the image
            
        Returns:
            Flipped or original image
        """
        if flip:
            return TF.hflip(img)
        return img


class RandomVerticalFlip(BaseRandomTransform):
    """Apply random vertical flip to a sequence of images."""
    
    def __init__(self, sequence_length: int, p: float = 0.5):
        """
        Initialize random vertical flip transform.
        
        Args:
            sequence_length: Length of the image sequence
            p: Probability of applying the flip
        """
        super().__init__(sequence_length)
        self.p = p
        
    def get_params(self) -> bool:
        """
        Determine whether to apply flip.
        
        Returns:
            True if flip should be applied, False otherwise
        """
        return random.random() < self.p
        
    def apply_transform(self, img: Union[Image.Image, torch.Tensor], flip: bool) -> Union[Image.Image, torch.Tensor]:
        """
        Apply vertical flip if indicated.
        
        Args:
            img: Input image
            flip: Whether to flip the image
            
        Returns:
            Flipped or original image
        """
        if flip:
            return TF.vflip(img)
        return img


# Aliases for backward compatibility
RandomBrightness.get_params = RandomBrightness.get_params
RandomContrast.get_params = RandomContrast.get_params
RandomGamma.get_params = RandomGamma.get_params
RandomHue.get_params = RandomHue.get_params
RandomSaturation.get_params = RandomSaturation.get_params
RandomHorizontalFlip.get_params = RandomHorizontalFlip.get_params
RandomVerticalFlip.get_params = RandomVerticalFlip.get_params
