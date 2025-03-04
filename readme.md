# Plant Transpiration Regression using Multi-Modal Imaging

A PyTorch-based framework for plant classification using multiple imaging modalities, including LWIR (Long-Wave Infrared) and VIR (Visible and Infrared) at various wavelengths.

## Overview

This project provides tools for loading, processing, and classifying plant images across different sensor modalities. The framework is designed to handle time series data with multiple plants and classes, enabling robust classification even in challenging conditions.

Key features:
- Support for multiple sensor modalities (LWIR, VIR at 577nm, 692nm, 732nm, 970nm, and polarized)
- Time-series data processing with temporal feature extraction
- Feature fusion across modalities for improved classification
- Data augmentation and preprocessing pipelines
- Visualization tools for model analysis

## Project Structure

```
project/
│
├── __init__.py                      # Main module exports
├── exceptions.py                    # Centralized exceptions
│
├── dataset/                         # Dataset-related components
│   ├── __init__.py
│   ├── labels.py                    # Label and position data
│   └── modalities.py                # Multi-modality dataset loaders
│
├── sensors/                         # Sensor-specific implementations
│   ├── __init__.py
│   ├── base.py                      # Base sensor class
│   ├── lwir.py                      # LWIR sensor
│   └── vir.py                       # VIR sensor and variants
│
├── model/                           # Model architectures
│   ├── __init__.py
│   ├── feature_extraction.py        # Feature extraction models
│   └── linear_wrapper.py            # Classification layer wrapper
│
├── training/                        # Training utilities
│   ├── __init__.py
│   ├── train_loop.py                # Main training loop
│   └── cluster.py                   # Clustering analysis tools
│
└── utils/                           # Utility functions
    ├── __init__.py
    ├── visualization.py             # Visualization utilities
    └── transformations.py           # Custom image transformations
```

## Sensor Modalities

The framework supports the following sensor modalities:

- **LWIR (Long-Wave Infrared)**: Thermal imaging that captures heat signatures
- **VIR (Visible and Infrared)** at multiple wavelengths:
  - 577nm: Green light absorption
  - 692nm: Red light absorption
  - 732nm: Near-infrared reflectance
  - 970nm: Water content indication
  - VIRPolar: Polarized light reflection

## Dataset Structure

The dataset consists of time-series images of plants organized by day. Each plant belongs to one of 6 classes ('A1', 'C', 'm82', 'P', '2d', '88'). The plants are arranged in a grid, and their positions are stored in the `labels.py` file.

### Classes and Labels

- 6 plant classes represented by labels 0-5
- 48 plants arranged in a grid (4 columns, 12 rows)
- Each plant has a specific position in the sensor field of view

## Model Architecture

The classification model consists of several components:

1. **ImageFeatureExtractor**: Uses a pre-trained InceptionV3 to extract features from individual frames
2. **ModalityFeatureExtractor**: Processes temporal sequences using Temporal Convolutional Networks (TCN)
3. **PlantFeatureExtractor**: Combines features from multiple modalities into a unified representation
4. **LinearWrapper**: Adds a classification layer on top of the feature extractor

## Usage Examples

### Loading the Dataset

```python
from datasets import Modalities
from datetime import datetime

# Initialize dataset with multiple modalities
dataset = Modalities(
    'path/to/experiment',
    'lwir', '577nm', '692nm',  # Modalities to include
    split_cycle=7,
    start_date=datetime(2019, 6, 4),
    end_date=datetime(2019, 6, 19)
)

# Split into training and testing sets
from datasets.modalities import ModalitiesSubset
train_set, test_set = ModalitiesSubset.random_split(dataset, [36, 12])
```

### Creating a Model

```python
from model import PlantFeatureExtractor
from model.linear_wrapper import LinearWrapper

# Create feature extractor with multiple modalities
feat_ext = PlantFeatureExtractor('lwir', '577nm', '692nm')

# Wrap with classification layer
model = LinearWrapper(feat_ext, embedding_dim=512, num_classes=6)
```

### Training the Model

```python
from training.train_loop import train_loop

# Train the model
train_loop(model, train_set, test_set, epochs=25)
```

### Visualizing Results

```python
import torch
from utils.visualization import visualize_modality, plot_confusion_matrix

# Visualize a sample from a specific modality
sample = dataset[0]
fig = visualize_modality(sample, 'lwir', plant_idx=0, time_idx=0)
fig.savefig('lwir_sample.png')

# Plot confusion matrix
y_true = torch.tensor([sample['label'] for sample in test_set])
y_pred = torch.argmax(model(test_set), dim=1)
fig = plot_confusion_matrix(y_true, y_pred, class_names=['A1', 'C', 'm82', 'P', '2d', '88'])
fig.savefig('confusion_matrix.png')
```

## Custom Transformations

The framework includes several custom transformations for image preprocessing:

- **RandomBrightness**: Adjusts image brightness consistently across sequences
- **RandomContrast**: Modifies image contrast
- **RandomGamma**: Applies gamma correction
- **RandomCrop**: Crops images to a specified size
- **RandomHorizontalFlip/RandomVerticalFlip**: Flips images with consistent parameters across sequences

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/plant-classification.git
cd plant-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

[Insert License Information]

## Acknowledgments

[Insert Acknowledgments]
