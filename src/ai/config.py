"""Centralized hyperparameters for src/ai/ V1 (PilotNet + speed)."""

from __future__ import annotations

# Image preprocessing
IMAGE_RAW_HEIGHT = 720
IMAGE_RAW_WIDTH = 1280
CROP_TOP_PX = 100
CROP_BOTTOM_PX = 60
IMAGE_HEIGHT = 88
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3

# Speed normalization (km/h)
SPEED_NORM_DIVISOR = 90.0

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-6

# Loss weights (steer dominant, CIL convention)
LOSS_WEIGHTS: dict[str, float] = {"steer": 1.0, "throttle": 0.5, "brake": 0.5}

# Data split
TRAIN_VAL_RATIO = 0.8
SHUFFLE_BUFFER = 1000

# Reproducibility
SEED = 42
