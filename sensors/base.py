"""
Base sensor class for plant image datasets.
"""
from datetime import datetime
import glob
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils import data
from torchvision import transforms

from ..dataset.labels import labels
from ..exceptions import DirEmptyError


class BaseSensor(data.Dataset):
    """Base class for all sensor data sources."""

    def __init__(
        self,
        root_dir: str,
        sensor_type: str,
        img_len: int,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        skip: int = 1,
        max_len: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the base sensor dataset.
        
        Args:
            root_dir: Path to the experiment directory
            sensor_type: Type of sensor ('lwir' or 'vir')
            img_len: Target image length/width for processing
            split_cycle: Number of days between data points 
            start_date: Start date for filtering data
            end_date: End date for filtering data
            skip: Number of frames to skip
            max_len: Maximum number of images to use (None for no limit)
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.sensor_type = sensor_type
        self.img_len = img_len
        self.split_cycle = split_cycle
        self.start_date = start_date
        self.end_date = end_date
        self.skip = skip
        self.max_len = max_len if max_len is not None else 10000
        
        # Set transforms
        self.transform = transform if transform is not None else transforms.Compose([])
        
        # To be set by child classes
        self.dir_glob_pattern = ""
        self.date_format = ""
        self.positions: List[Tuple[int, int]] = []
        
    def _filter_dirs(self, dirs: List[str]) -> List[str]:
        """
        Filter directories based on date range.
        
        Args:
            dirs: List of directory paths
            
        Returns:
            List of filtered directory paths
        """
        filtered = []
        for dir_path in dirs:
            try:
                date = datetime.strptime(dir_path, self.date_format)
                if self.start_date <= date <= self.end_date:
                    filtered.append(dir_path)
            except ValueError:
                # Skip directories that don't match the expected format
                continue
                
        return filtered
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.positions) * self.split_cycle
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, Tuple[int, int]]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image, label, position, and plant index
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")
            
        # Calculate which day in the cycle and which plant
        cycle_day = idx // len(self.positions)
        plant = idx % len(self.positions)
        
        # Process and return sample
        sample = self._get_sample(cycle_day, plant)
        
        return sample
    
    def _get_sample(self, cycle_day: int, plant: int) -> Dict[str, Union[torch.Tensor, int, Tuple[int, int]]]:
        """
        Create a sample for a specific cycle day and plant.
        
        Args:
            cycle_day: Day in the cycle 
            plant: Plant index
            
        Returns:
            Sample dictionary
        """
        raise NotImplementedError("Subclasses must implement _get_sample")
    
    def _get_day(self, dir_path: str) -> str:
        """
        Extract the day from a directory path.
        
        Args:
            dir_path: Directory path
            
        Returns:
            Day as a string
        """
        # Extract only the filename part
        basename = os.path.basename(dir_path)
        # Split by underscore and get the day part (3rd component, index 2)
        return basename.split('_')[2]
