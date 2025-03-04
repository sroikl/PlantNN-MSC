"""
LWIR sensor data loader for plant image datasets.
"""
from datetime import datetime
import glob
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

from .base import BaseSensor
from ..dataset.labels import labels, lwir_positions
from ..exceptions import DirEmptyError


class LWIR(BaseSensor):
    """
    Long-Wave Infrared (LWIR) data loader.
    
    Loads and processes LWIR thermal images from the experiment directories.
    """

    def __init__(
        self,
        root_dir: str,
        img_len: int = 229,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        skip: int = 1,
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the LWIR dataset.
        
        Args:
            root_dir: Path to the experiment directory
            img_len: Target image size after resizing
            split_cycle: Number of days between data points
            start_date: Start date for filtering data
            end_date: End date for filtering data
            skip: Number of frames to skip
            max_len: Maximum number of images to use (None for no limit)
            transform: Optional transforms to apply to tensor images
        """
        super().__init__(
            root_dir=root_dir,
            sensor_type='lwir',
            img_len=img_len,
            split_cycle=split_cycle,
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            max_len=max_len,
            transform=transform
        )
        
        # Set LWIR-specific parameters
        self.positions = lwir_positions
        self.plant_crop_len = 60
        self.dir_glob_pattern = os.path.join(root_dir, '*LWIR')
        self.date_format = f"{root_dir}/%Y_%m_%d_%H_%M_%S_LWIR"
        
        # Get and filter LWIR directories
        self.lwir_dirs = sorted(glob.glob(self.dir_glob_pattern))[::skip]
        self.lwir_dirs = self._filter_dirs(self.lwir_dirs)
        
        if not self.lwir_dirs:
            raise DirEmptyError(f"No LWIR directories found in {root_dir} matching date range")

    def _get_sample(self, cycle_day: int, plant: int) -> Dict[str, Union[torch.Tensor, int, Tuple[int, int]]]:
        """
        Create a sample for a specific cycle day and plant.
        
        Args:
            cycle_day: Day in the cycle
            plant: Plant index
            
        Returns:
            Sample dictionary with image, label, position, and plant index
        """
        to_tensor = transforms.ToTensor()
        
        tensors = []
        cur_day = self._get_day(self.lwir_dirs[0])
        remaining_days = cycle_day
        
        for lwir_dir in self.lwir_dirs:
            # Update the current day when it changes
            if cur_day != self._get_day(lwir_dir):
                cur_day = self._get_day(lwir_dir)
                remaining_days -= 1
            
            # Process images only on the specific cycle days
            if remaining_days % self.split_cycle != 0:
                continue
                
            try:
                image = self._get_image(lwir_dir, plant)
                tensors.append(to_tensor(image).float())
            except DirEmptyError:
                continue
                
        # Apply limits and transformations
        tensors = tensors[:self.max_len]
        tensors = [self.transform(tensor) for tensor in tensors]
        
        if not tensors:
            # Create an empty tensor with the right dimensions if no valid images found
            image = torch.zeros((1, self.img_len, self.img_len))
        else:
            image = torch.cat(tensors)
            
        return {
            'image': image, 
            'label': labels[plant],
            'position': self.positions[plant], 
            'plant': plant
        }
        
    def _get_image(self, lwir_dir: str, plant_idx: int) -> Image.Image:
        """
        Extract and preprocess the plant image from the LWIR data.
        
        Args:
            lwir_dir: LWIR directory path
            plant_idx: Plant index
            
        Returns:
            Processed PIL Image
            
        Raises:
            DirEmptyError: If no TIFF images found in the directory
        """
        pos = self.positions[plant_idx]
        
        # Calculate crop boundaries
        left = pos[0] - self.plant_crop_len // 2
        right = pos[0] + self.plant_crop_len // 2
        top = pos[1] - self.plant_crop_len // 2
        bottom = pos[1] + self.plant_crop_len // 2
        
        # Find image file
        image_paths = glob.glob(os.path.join(lwir_dir, '*.tiff'))
        if not image_paths:
            raise DirEmptyError(f"No TIFF images found in {lwir_dir}")
            
        image_path = image_paths[0]
        
        # Process image
        image = Image.open(image_path)
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.img_len, self.img_len))
        
        return image
