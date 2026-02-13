"""
VIPR Classification Framework

Utilities for training and evaluating image classification models.
"""

from .model_utils import load_model, count_parameters, BACKBONE_REGISTRY
from .trainer import train_model
