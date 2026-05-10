"""PilotNet + speed conditioning, V1 of the central AI."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.ai.config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


def build_pilotnet_speed() -> keras.Model:
    """NVIDIA PilotNet backbone + speed scalar concat + 3 named regression heads.

    Inputs:
        image: (B, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS) float32 in [0, 1]
        speed: (B, 1) float32 normalized km/h (e.g. /90)

    Outputs (dict):
        steer:    (B, 1) tanh    in [-1, 1]
        throttle: (B, 1) sigmoid in [0, 1]
        brake:    (B, 1) sigmoid in [0, 1]
    """
    image_in = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        name="image",
    )
    speed_in = keras.Input(shape=(1,), name="speed")

    x = layers.Conv2D(24, 5, strides=2, activation="relu")(image_in)
    x = layers.Conv2D(36, 5, strides=2, activation="relu")(x)
    x = layers.Conv2D(48, 5, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Flatten()(x)

    fused = layers.Concatenate()([x, speed_in])
    fused = layers.Dense(100, activation="relu")(fused)
    fused = layers.Dropout(0.2)(fused)
    fused = layers.Dense(50, activation="relu")(fused)
    fused = layers.Dropout(0.2)(fused)
    fused = layers.Dense(10, activation="relu")(fused)

    steer = layers.Dense(1, activation="tanh", name="steer")(fused)
    throttle = layers.Dense(1, activation="sigmoid", name="throttle")(fused)
    brake = layers.Dense(1, activation="sigmoid", name="brake")(fused)

    return keras.Model(
        inputs={"image": image_in, "speed": speed_in},
        outputs={"steer": steer, "throttle": throttle, "brake": brake},
        name="pilotnet_speed",
    )
