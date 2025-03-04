"""
VIR (Visible and Infrared) sensor data loaders for plant image datasets.
"""
from datetime import datetime
import glob
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from .base import BaseSensor
from ..dataset.labels import labels, vir_positions, vir_img_size
from ..exceptions import DirEmptyError


class VIR(BaseSensor):
    """
    Base class for all Visible and Infrared (VIR) data loaders.
    
    Provides methods for loading and processing VIR images at different wavelengths.
    """

    def __init__(
        self,
        root_dir: str,
        vir_type: str,
        img_len: int = 458,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the VIR dataset.
        
        Args:
            root_dir: Path to the experiment directory
            vir_type: Type of VIR images (e.g. "577nm", "692nm")
            img_len: Target image size
            split_cycle: Number of days between data points
            start_date: Start date for filtering data
            end_date: End date for filtering data
            max_len: Maximum number of images to use (None for no limit)
            transform: Optional transforms to apply to tensor images
        """
        super().__init__(
            root_dir=root_dir,
            sensor_type='vir',
            img_len=img_len,
            split_cycle=split_cycle,
            start_date=start_date,
            end_date=end_date,
            skip=1,  # VIR doesn't use skip parameter
            max_len=max_len,
            transform=transform
        )
        
        # Set VIR-specific parameters
        self.positions = vir_positions
        self.vir_type = vir_type
        self.dir_glob_pattern = os.path.join(root_dir, '*VIR_day')
        self.date_format = f"{root_dir}/%Y_%m_%d_%H_%M_%S_VIR_day"
        
        # Get and filter VIR directories
        self.vir_dirs = sorted(glob.glob(self.dir_glob_pattern))
        self.vir_dirs = self._filter_dirs(self.vir_dirs)
        
        if not self.vir_dirs:
            raise DirEmptyError(f"No VIR directories found in {root_dir} matching date range")

    def _get_sample(self, cycle_day: int, plant: int) -> Dict[str, Union[torch.Tensor, int, Tuple[int, int]]]:
        """
        Create a sample for a specific cycle day and plant.
        
        Args:
            cycle_day: Day in the cycle
            plant: Plant index
            
        Returns:
            Sample dictionary with image, label, position, and plant index
        """
        tensors = []
        cur_day = self._get_day(self.vir_dirs[0])
        remaining_days = cycle_day
        
        for vir_dir in self.vir_dirs:
            # Update the current day when it changes
            if cur_day != self._get_day(vir_dir):
                cur_day = self._get_day(vir_dir)
                remaining_days -= 1
                
            # Process images only on the specific cycle days
            if remaining_days % self.split_cycle != 0:
                continue
                
            try:
                arr = self._get_np_arr(vir_dir, plant)
                tensor = torch.from_numpy(arr).float()
                tensor.unsqueeze_(0)  # Add channel dimension
                tensors.append(tensor)
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
    
    def _get_np_arr(self, vir_dir: str, plant_idx: int) -> np.ndarray:
        """
        Extract and preprocess the plant image from the VIR data.
        
        Args:
            vir_dir: VIR directory path
            plant_idx: Plant index
            
        Returns:
            Processed numpy array
            
        Raises:
            DirEmptyError: If no RAW images found in the directory for the given VIR type
        """
        pos = self.positions[plant_idx]
        
        # Calculate crop boundaries
        left = pos[0] - self.img_len // 2
        right = pos[0] + self.img_len // 2
        top = pos[1] - self.img_len // 2
        bottom = pos[1] + self.img_len // 2
        
        # Find image file
        image_path = glob.glob(f"{vir_dir}/*{self.vir_type}*.raw")
        if not image_path:
            raise DirEmptyError(f"No {self.vir_type} RAW images found in {vir_dir}")
            
        image_path = image_path[0]
        exposure = self._get_exposure(image_path)
        
        # Process image
        arr = np.fromfile(image_path, dtype=np.int16).reshape(vir_img_size[1], vir_img_size[0])
        arr = arr[top:bottom, left:right] / exposure
        
        return arr
        
    @staticmethod
    def _get_exposure(file_name: str) -> float:
        """
        Extract exposure time from the filename.
        
        Args:
            file_name: RAW file name
            
        Returns:
            Exposure time as a float
        """
        return float(file_name.split('ET')[-1].split('.')[0])


class VIR577nm(VIR):
    """VIR data loader for 577nm wavelength."""
    
    def __init__(
        self,
        root_dir: str,
        img_len: int = 458,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(
            root_dir=root_dir,
            vir_type="577nm",
            img_len=img_len,
            split_cycle=split_cycle,
            start_date=start_date,
            end_date=end_date,
            max_len=max_len,
            transform=transform
        )


class VIR692nm(VIR):
    """VIR data loader for 692nm wavelength."""
    
    def __init__(
        self,
        root_dir: str,
        img_len: int = 458,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(
            root_dir=root_dir,
            vir_type="692nm",
            img_len=img_len,
            split_cycle=split_cycle,
            start_date=start_date,
            end_date=end_date,
            max_len=max_len,
            transform=transform
        )


class VIR732nm(VIR):
    """VIR data loader for 732nm wavelength."""
    
    def __init__(
        self,
        root_dir: str,
        img_len: int = 458,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(
            root_dir=root_dir,
            vir_type="732nm",
            img_len=img_len,
            split_cycle=split_cycle,
            start_date=start_date,
            end_date=end_date,
            max_len=max_len,
            transform=transform
        )


class VIR970nm(VIR):
    """VIR data loader for 970nm wavelength."""
    
    def __init__(
        self,
        root_dir: str,
        img_len: int = 458,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(
            root_dir=root_dir,
            vir_type="970nm",
            img_len=img_len,
            split_cycle=split_cycle,
            start_date=start_date,
            end_date=end_date,
            max_len=max_len,
            transform=transform
        )


class VIRPolar(VIR):
    """VIR data loader for Polarizer type."""
    
    def __init__(
        self,
        root_dir: str,
        img_len: int = 458,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__(
            root_dir=root_dir,
            vir_type="Polarizer",
            img_len=img_len,
            split_cycle=split_cycle,
            start_date=start_date,
            end_date=end_date,
            max_len=max_len,
            transform=transform
        )
