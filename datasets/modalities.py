"""
Modalities dataset classes for combining multiple sensor types.
"""
from datetime import datetime
import random
from typing import Dict, List, Optional, Tuple, Union, Any

from torch.utils import data

from ..sensors.lwir import LWIR
from ..sensors.vir import VIR577nm, VIR692nm, VIR732nm, VIR970nm, VIRPolar
from ..dataset.labels import labels
from ..exceptions import InvalidModalityError, ConfigurationError


# Mapping of modality names to their respective classes
MODALITY_MAP = {
    'lwir': LWIR,
    '577nm': VIR577nm,
    '692nm': VIR692nm,
    '732nm': VIR732nm,
    '970nm': VIR970nm,
    'polar': VIRPolar
}


class Modalities(data.Dataset):
    """
    Dataset for combining multiple modalities (sensor types).
    
    This class allows loading and combining different sensor modalities
    for plant classification tasks.
    """

    def __init__(
        self,
        root_dir: str,
        *mods: str,
        split_cycle: int = 7,
        start_date: datetime = datetime(2019, 6, 4),
        end_date: datetime = datetime(2019, 7, 7),
        transform: Optional[Any] = None,
        **k_mods: Dict
    ):
        """
        Initialize the multi-modality dataset.
        
        Args:
            root_dir: Path to the experiment directory
            *mods: Modality names to load with default parameters
            split_cycle: Number of days between data points
            start_date: Start date for filtering data
            end_date: End date for filtering data
            transform: Optional transform to apply to samples
            **k_mods: Modality names with custom initialization parameters
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split_cycle = split_cycle
        self.num_plants = len(labels)
        
        # Initialize modalities dictionary
        self.modalities = {}
        
        # If no modalities specified, use all available ones
        if not mods and not k_mods:
            mods = tuple(MODALITY_MAP.keys())
            
        # Initialize modalities with default parameters
        for mod in mods:
            if mod not in MODALITY_MAP:
                raise InvalidModalityError(f"Unknown modality: {mod}")
                
            self.modalities[mod] = MODALITY_MAP[mod](
                root_dir=root_dir,
                split_cycle=split_cycle,
                start_date=start_date,
                end_date=end_date
            )
            
        # Initialize modalities with custom parameters
        for mod, params in k_mods.items():
            if mod not in MODALITY_MAP:
                raise InvalidModalityError(f"Unknown modality: {mod}")
                
            self.modalities[mod] = MODALITY_MAP[mod](
                root_dir=root_dir,
                split_cycle=split_cycle,
                start_date=start_date,
                end_date=end_date,
                **params
            )
            
        if not self.modalities:
            raise ConfigurationError("No modalities specified or found")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        # All modalities have the same length, so we can use any of them
        dataset = next(iter(self.modalities.values()))
        return len(dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with all modalities at the given index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with modality data, label, and plant index
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")
            
        # Get the plant index
        plant = idx % self.num_plants
        
        # Collect all modality data for this index
        sample = {
            mod: dataset[idx]['image'] for mod, dataset in self.modalities.items()
        }
        
        # Add metadata
        sample['label'] = labels[plant]
        sample['plant'] = plant
        
        # Apply transform if available
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class ModalitiesSubset(data.Dataset):
    """
    Subset of a Modalities dataset for specific plants.
    
    This class allows creating subsets of the full dataset for
    particular plants, useful for training/validation splits.
    """
    
    def __init__(self, modalities: Modalities, plants: List[int]):
        """
        Initialize a subset of the modalities dataset.
        
        Args:
            modalities: Parent Modalities dataset
            plants: List of plant indices to include in the subset
        """
        self.data = modalities
        self.split_cycle = modalities.split_cycle
        self.plants = plants
        self.num_plants = len(plants)
        
        if not plants:
            raise ConfigurationError("Empty plant list for ModalitiesSubset")

    def __len__(self) -> int:
        """Return the total number of samples in the subset."""
        return self.num_plants * self.split_cycle

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the subset.
        
        Args:
            idx: Index in the subset
            
        Returns:
            Dictionary with modality data, label, and local plant index
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")
            
        # Get the plant and cycle indices
        local_plant_idx = idx % self.num_plants
        cycle = idx // self.num_plants
        
        # Map to the global plant index in the full dataset
        global_plant_idx = self.plants[local_plant_idx]
        
        # Calculate the index in the full dataset
        full_dataset_idx = self.data.num_plants * cycle + global_plant_idx
        
        # Get the data from the full dataset
        data = self.data[full_dataset_idx]
        
        # Update the plant index to the local index
        data['plant'] = local_plant_idx
        
        return data

    @staticmethod
    def random_split(modalities: Modalities, plants_amounts: List[int]) -> List['ModalitiesSubset']:
        """
        Randomly split the dataset into multiple subsets.
        
        Args:
            modalities: Parent Modalities dataset
            plants_amounts: List of number of plants for each subset
            
        Returns:
            List of ModalitiesSubset objects
        """
        # Validate total plants
        if sum(plants_amounts) > modalities.num_plants:
            raise ConfigurationError(
                f"Sum of plant amounts ({sum(plants_amounts)}) exceeds "
                f"total plants ({modalities.num_plants})"
            )
            
        # Random shuffle of all plant indices
        all_plant_indices = list(range(modalities.num_plants))
        random.shuffle(all_plant_indices)
        
        # Create subsets
        subsets = []
        start_idx = 0
        
        for amount in plants_amounts:
            end_idx = start_idx + amount
            subset_plants = all_plant_indices[start_idx:end_idx]
            subsets.append(ModalitiesSubset(modalities, subset_plants))
            start_idx = end_idx
            
        return subsets

    @staticmethod
    def leave_one_out(modalities: Modalities, plant_idx: int) -> Tuple['ModalitiesSubset', 'ModalitiesSubset']:
        """
        Create leave-one-out split (one plant for testing, rest for training).
        
        Args:
            modalities: Parent Modalities dataset
            plant_idx: Index of the plant to leave out
            
        Returns:
            Tuple of (test_subset, train_subset)
        """
        if plant_idx < 0 or plant_idx >= modalities.num_plants:
            raise IndexError(f"Plant index {plant_idx} out of bounds")
            
        # Create indices for the rest of the plants
        rest_idx = list(range(modalities.num_plants))
        rest_idx.pop(plant_idx)
        
        # Create the two subsets
        test_subset = ModalitiesSubset(modalities, [plant_idx])
        train_subset = ModalitiesSubset(modalities, rest_idx)
        
        return test_subset, train_subset