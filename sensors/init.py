"""
Plant image dataset framework for multiple modalities.
"""

from .exceptions import DirEmptyError
from .sensors.lwir import LWIR
from .sensors.vir import VIR577nm, VIR692nm, VIR732nm, VIR970nm, VIRPolar
from .dataset.modalities import Modalities, ModalitiesSubset
from .dataset.labels import classes, labels, positions

__all__ = [
    # Sensors
    'LWIR', 'VIR577nm', 'VIR692nm', 'VIR732nm', 'VIR970nm', 'VIRPolar',
    # Dataset components
    'Modalities', 'ModalitiesSubset',
    # Data structures
    'classes', 'labels', 'positions',
    # Exceptions
    'DirEmptyError'
]
