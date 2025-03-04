"""
Visualization utilities for plant classification results and data.

This module provides functions for visualizing:
- Raw sensor data from different modalities
- Model predictions and classifications
- Performance metrics and training history
- Feature embeddings and activations
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageDraw, ImageFont
import io
from matplotlib.colors import LinearSegmentedColormap


def visualize_modality(
    sample: Dict[str, torch.Tensor],
    modality: str,
    plant_idx: int = 0,
    time_idx: int = 0,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    normalize: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a single modality image for a specific plant and time index.
    
    Args:
        sample: Dictionary containing modality data
        modality: Modality name to visualize (e.g., 'lwir', '577nm')
        plant_idx: Plant index to visualize
        time_idx: Time index to visualize
        figsize: Figure size
        cmap: Colormap to use
        normalize: Whether to normalize the image
        title: Custom title (default: uses modality name)
        
    Returns:
        Matplotlib figure
    """
    if modality not in sample:
        raise ValueError(f"Modality '{modality}' not found in sample. Available: {list(sample.keys())}")
    
    # Extract the image data
    if isinstance(sample[modality], dict) and 'image' in sample[modality]:
        image_data = sample[modality]['image']
    else:
        image_data = sample[modality]
    
    # Handle multi-dimensional data
    if len(image_data.shape) > 3:
        if time_idx >= image_data.shape[1]:
            raise ValueError(f"Time index {time_idx} out of bounds (max: {image_data.shape[1]-1})")
        image = image_data[plant_idx, time_idx]
    else:
        image = image_data[plant_idx]
    
    # Convert to numpy and squeeze if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    
    # Normalize if requested
    if normalize:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set title
    if title is None:
        if 'label' in sample and plant_idx < len(sample['label']):
            class_label = sample['label'][plant_idx].item() if isinstance(sample['label'], torch.Tensor) else sample['label'][plant_idx]
            title = f"{modality.upper()} - Plant {plant_idx} (Class {class_label}) - Time {time_idx}"
        else:
            title = f"{modality.upper()} - Plant {plant_idx} - Time {time_idx}"
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def visualize_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    top_k: int = 3,
    figsize: Tuple[int, int] = (12, 6),
    sample_indices: Optional[List[int]] = None
) -> plt.Figure:
    """
    Visualize model predictions compared to ground truth.
    
    Args:
        predictions: Model prediction logits or probabilities
        targets: Ground truth labels
        class_names: List of class names
        top_k: Number of top predictions to show
        figsize: Figure size
        sample_indices: Specific sample indices to visualize (default: first 5)
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Apply softmax if needed (if logits)
    if np.min(predictions) < 0 or np.max(predictions) > 1:
        predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    
    # Sample selection
    if sample_indices is None:
        sample_indices = list(range(min(5, len(predictions))))
    
    num_samples = len(sample_indices)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=figsize, squeeze=False)
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i, 0]
        
        # Get prediction data
        pred = predictions[idx]
        target = targets[idx]
        
        # Get top k predictions
        top_k_indices = np.argsort(pred)[::-1][:top_k]
        top_k_values = pred[top_k_indices]
        top_k_names = [class_names[j] for j in top_k_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(top_k_names, top_k_values, color='skyblue')
        
        # Highlight ground truth
        for j, (name, value, idx_) in enumerate(zip(top_k_names, top_k_values, top_k_indices)):
            if idx_ == target:
                bars[j].set_color('green')
                ax.text(value + 0.01, j, '✓', va='center', fontweight='bold', color='green')
        
        # Check if ground truth is not in top k
        if target not in top_k_indices:
            target_class = class_names[target]
            target_value = pred[target]
            ax.barh([target_class], [target_value], color='red')
            ax.text(target_value + 0.01, top_k, '✓', va='center', fontweight='bold', color='red')
        
        # Set title and labels
        ax.set_title(f"Sample {idx}")
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1.1)
        
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    class_names: List[str],
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix'
) -> plt.Figure:
    """
    Plot a confusion matrix for classification results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize by row (true label)
        figsize: Figure size
        cmap: Colormap to use
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    metrics: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary of metrics (keys) and their values over epochs (values)
        figsize: Figure size
        metrics: Specific metrics to plot (default: all except validation metrics)
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        # Default to all metrics except validation ones
        metrics = [key for key in history.keys() if not key.startswith('val_')]
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training metric
        ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        # Set labels and legend
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Only set xlabel for the bottom subplot
        if i == len(metrics) - 1:
            ax.set_xlabel('Epoch')
    
    plt.tight_layout()
    return fig


def visualize_multi_modality(
    sample: Dict[str, torch.Tensor],
    modalities: List[str],
    plant_idx: int = 0,
    time_idx: int = 0,
    figsize: Tuple[int, int] = (16, 10),
    cmaps: Optional[Dict[str, str]] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Visualize multiple modalities for a single plant and time point.
    
    Args:
        sample: Dictionary containing modality data
        modalities: List of modality names to visualize
        plant_idx: Plant index to visualize
        time_idx: Time index to visualize
        figsize: Figure size
        cmaps: Dictionary mapping modality names to colormaps
        normalize: Whether to normalize each image
        
    Returns:
        Matplotlib figure
    """
    # Validate modalities
    available_mods = [mod for mod in modalities if mod in sample]
    if not available_mods:
        raise ValueError(f"None of the requested modalities {modalities} found in sample. Available: {list(sample.keys())}")
    
    # Create figure
    n_mods = len(available_mods)
    cols = min(3, n_mods)
    rows = (n_mods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    # Set default colormaps if not provided
    if cmaps is None:
        cmaps = {
            'lwir': 'hot',
            '577nm': 'Greens',
            '692nm': 'Reds',
            '732nm': 'Oranges',
            '970nm': 'Purples',
            'polar': 'viridis'
        }
        # Set default for any other modality
        for mod in available_mods:
            if mod not in cmaps:
                cmaps[mod] = 'viridis'
    
    # Plot each modality
    for i, modality in enumerate(available_mods):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Extract the image data
        if isinstance(sample[modality], dict) and 'image' in sample[modality]:
            image_data = sample[modality]['image']
        else:
            image_data = sample[modality]
        
        # Handle multi-dimensional data
        if len(image_data.shape) > 3:
            image = image_data[plant_idx, time_idx]
        else:
            image = image_data[plant_idx]
        
        # Convert to numpy and squeeze
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.squeeze(image)
        
        # Normalize if requested
        if normalize:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Plot the image
        cmap = cmaps.get(modality, 'viridis')
        im = ax.imshow(image, cmap=cmap)
        plt.colorbar(im, ax=ax)
        
        # Set title
        ax.set_title(f"{modality.upper()}")
    
    # Hide unused subplots
    for i in range(len(available_mods), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    # Create overall title
    if 'label' in sample and plant_idx < len(sample['label']):
        class_label = sample['label'][plant_idx].item() if isinstance(sample['label'], torch.Tensor) else sample['label'][plant_idx]
        plt.suptitle(f"Plant {plant_idx} (Class {class_label}) - Time {time_idx}", fontsize=16)
    else:
        plt.suptitle(f"Plant {plant_idx} - Time {time_idx}", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def create_feature_activation_map(
    model: nn.Module,
    sample: Dict[str, torch.Tensor],
    target_modality: str,
    plant_idx: int = 0,
    time_idx: int = 0,
    layer_name: Optional[str] = None,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'jet'
) -> plt.Figure:
    """
    Create a feature activation map overlaid on the original image.
    
    Note: This is a simplified implementation and requires model to have hooks registered
    or be designed for activation extraction.
    
    Args:
        model: Model to extract activations from
        sample: Dictionary containing modality data
        target_modality: Modality to visualize
        plant_idx: Plant index to visualize
        time_idx: Time index to visualize
        layer_name: Name of layer to extract activations from (None for last layer)
        alpha: Opacity of the activation map overlay
        figsize: Figure size
        cmap: Colormap for activation map
        
    Returns:
        Matplotlib figure
    """
    # This is a placeholder implementation that would need to be customized
    # based on the specific model architecture and how activations are extracted
    
    # Extract the image data
    if isinstance(sample[target_modality], dict) and 'image' in sample[target_modality]:
        image_data = sample[target_modality]['image']
    else:
        image_data = sample[target_modality]
    
    # Get the image of interest
    if len(image_data.shape) > 3:
        image = image_data[plant_idx, time_idx]
    else:
        image = image_data[plant_idx]
    
    # Convert to numpy and squeeze
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.squeeze(image)
    
    # Normalize image for display
    normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Create a mock activation map (replace with actual extraction)
    # In practice, you would run the model and extract activations using hooks
    mock_activation = np.random.rand(*image.shape)  # Replace with actual activation
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display original image
    ax.imshow(normalized_image, cmap='gray')
    
    # Overlay activation map
    activation_map = ax.imshow(mock_activation, cmap=cmap, alpha=alpha)
    
    # Add colorbar
    plt.colorbar(activation_map, ax=ax)
    
    # Set title
    ax.set_title(f"Activation Map for {target_modality.upper()} - Plant {plant_idx}, Time {time_idx}")
    
    plt.tight_layout()
    return fig


def visualize_embedding_space(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    method: str = 'tsne',
    perplexity: int = 30,
    figsize: Tuple[int, int] = (12, 10),
    marker_size: int = 50,
    title: str = 'Feature Embedding Space'
) -> plt.Figure:
    """
    Visualize the embedding space using dimensionality reduction.
    
    Args:
        embeddings: Feature embeddings (N x D)
        labels: Class labels for each embedding
        class_names: List of class names
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        perplexity: Perplexity parameter for t-SNE
        figsize: Figure size
        marker_size: Size of scatter plot markers
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Check dimensionality
    if embeddings.shape[1] <= 2:
        # Already 2D or 1D, no need for reduction
        reduced_data = embeddings
        if embeddings.shape[1] == 1:
            # If 1D, add a zero second dimension
            reduced_data = np.column_stack((reduced_data, np.zeros_like(reduced_data)))
    else:
        # Perform dimensionality reduction
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        elif method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        reduced_data = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot for each class
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        idx = labels == label
        ax.scatter(
            reduced_data[idx, 0],
            reduced_data[idx, 1],
            s=marker_size,
            label=class_names[label] if label < len(class_names) else f"Class {label}",
            alpha=0.7
        )
    
    # Add legend and labels
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel(f"Dimension 1")
    ax.set_ylabel(f"Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def visualize_modality_time_series(
    sample: Dict[str, torch.Tensor],
    modality: str,
    plant_idx: int = 0,
    time_indices: Optional[List[int]] = None,
    num_timepoints: int = 5,
    figsize: Tuple[int, int] = (16, 4),
    cmap: str = 'viridis',
    normalize: bool = True
) -> plt.Figure:
    """
    Visualize a time series of images for a specific modality and plant.
    
    Args:
        sample: Dictionary containing modality data
        modality: Modality name to visualize
        plant_idx: Plant index to visualize
        time_indices: Specific time indices to visualize (default: evenly spaced)
        num_timepoints: Number of timepoints to show (if time_indices not provided)
        figsize: Figure size
        cmap: Colormap to use
        normalize: Whether to normalize each image
        
    Returns:
        Matplotlib figure
    """
    if modality not in sample:
        raise ValueError(f"Modality '{modality}' not found in sample. Available: {list(sample.keys())}")
    
    # Extract the image data
    if isinstance(sample[modality], dict) and 'image' in sample[modality]:
        image_data = sample[modality]['image']
    else:
        image_data = sample[modality]
    
    # Handle different data structures
    if len(image_data.shape) <= 3:
        raise ValueError(f"Modality '{modality}' does not have a time dimension")
    
    # Determine time indices to visualize
    if time_indices is None:
        max_time = image_data.shape[1]
        if num_timepoints >= max_time:
            time_indices = list(range(max_time))
        else:
            # Evenly space the time indices
            time_indices = [int(i * (max_time - 1) / (num_timepoints - 1)) for i in range(num_timepoints)]
    
    # Create figure
    n_times = len(time_indices)
    fig, axes = plt.subplots(1, n_times, figsize=figsize)
    if n_times == 1:
        axes = [axes]
    
    # Get global min/max for consistent normalization across time
    if normalize:
        images = [image_data[plant_idx, t].detach().cpu().numpy() if isinstance(image_data, torch.Tensor) 
                 else image_data[plant_idx, t] for t in time_indices]
        images = [np.squeeze(img) for img in images]
        vmin = min(img.min() for img in images)
        vmax = max(img.max() for img in images)
    
    # Plot each timepoint
    for i, time_idx in enumerate(time_indices):
        ax = axes[i]
        
        # Get the image for this timepoint
        image = image_data[plant_idx, time_idx]
        
        # Convert to numpy and squeeze
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.squeeze(image)
        
        # Normalize if requested
        if normalize:
            image = (image - vmin) / (vmax - vmin + 1e-8)
        
        # Plot image
        im = ax.imshow(image, cmap=cmap)
        
        # Set title
        ax.set_title(f"Time {time_idx}")
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar at the right side of the figure
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    
    # Add overall title
    if 'label' in sample and plant_idx < len(sample['label']):
        class_label = sample['label'][plant_idx].item() if isinstance(sample['label'], torch.Tensor) else sample['label'][plant_idx]
        plt.suptitle(f"{modality.upper()} - Plant {plant_idx} (Class {class_label}) - Time Series", fontsize=16)
    else:
        plt.suptitle(f"{modality.upper()} - Plant {plant_idx} - Time Series", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    return fig
