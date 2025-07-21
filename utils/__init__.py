"""
Initialisation du package utils.
"""

from .data_utils import DataProcessor, load_dataset_config, get_dataset_statistics
from .model_utils import (
    ModelManager,
    PredictionPostProcessor,
    load_pretrained_model,
    get_available_models,
)
from .visualization import (
    MedicalImageVisualizer,
    plot_training_metrics,
    create_medical_image_montage,
)
from .metrics import (
    DetectionMetrics,
    ClassificationMetrics,
    create_metrics_dashboard,
    save_metrics_report,
)

__version__ = "1.0.0"

__all__ = [
    # Data utilities
    "DataProcessor",
    "load_dataset_config",
    "get_dataset_statistics",
    # Model utilities
    "ModelManager",
    "PredictionPostProcessor",
    "load_pretrained_model",
    "get_available_models",
    # Visualization utilities
    "MedicalImageVisualizer",
    "plot_training_metrics",
    "create_medical_image_montage",
    # Metrics utilities
    "DetectionMetrics",
    "ClassificationMetrics",
    "create_metrics_dashboard",
    "save_metrics_report",
]
