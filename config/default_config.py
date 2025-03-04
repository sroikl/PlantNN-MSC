"""
Default configuration parameters for the plant classification project.
"""
from typing import Dict, Any, List

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    # Feature extraction parameters
    "embedding_size": 512,
    "modality_embedding_size": 128,
    
    # TCN parameters
    "num_levels": 3,
    "num_hidden": 600,
    "kernel_size": 2,
    "dropout": 0.2,
    
    # Training parameters
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 16,
}

# Default training configuration
DEFAULT_TRAINING_CONFIG = {
    "epochs": 100,
    "early_stopping_patience": 10,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
    "gradient_clip_val": 1.0,
    "seed": 42,
    "val_check_interval": 1.0,
    "log_every_n_steps": 10,
}

# Modality-specific configurations
MODALITY_CONFIGS = {
    # LWIR modality configuration
    "lwir": {
        "num_levels": 3,
        "num_hidden": 600,
        "embedding_size": 128,
        "kernel_size": 2,
        "dropout": 0.2,
    },
    
    # VIR577nm modality configuration
    "577nm": {
        "num_levels": 3,
        "num_hidden": 600,
        "embedding_size": 128,
        "kernel_size": 2,
        "dropout": 0.2,
    },
    
    # VIR692nm modality configuration
    "692nm": {
        "num_levels": 3,
        "num_hidden": 600,
        "embedding_size": 128,
        "kernel_size": 2,
        "dropout": 0.2,
    },
    
    # VIR732nm modality configuration
    "732nm": {
        "num_levels": 3,
        "num_hidden": 600,
        "embedding_size": 128,
        "kernel_size": 2,
        "dropout": 0.2,
    },
    
    # VIR970nm modality configuration
    "970nm": {
        "num_levels": 3,
        "num_hidden": 600,
        "embedding_size": 128,
        "kernel_size": 2,
        "dropout": 0.2,
    },
    
    # VIRPolar modality configuration
    "polar": {
        "num_levels": 3,
        "num_hidden": 600,
        "embedding_size": 128,
        "kernel_size": 2,
        "dropout": 0.2,
    },
}
