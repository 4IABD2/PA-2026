from __future__ import annotations

import colorsys

import numpy as np

CAMERA_LOCATION = (0.30, 0.0, 1.50)
CAMERA_ROTATION_PITCH = -5.0


def decode_carla_depth(rgb: np.ndarray, max_depth_m: float = 1000.0) -> np.ndarray:
    """Décode les 3 canaux RGB en distance (mètres).

    Formule CARLA : ``meters = ((R + G*256 + B*256**2) / (256**3 - 1)) * 1000``.
    Plage physique 0-1000 m ; on clippe à ``max_depth_m`` pour le ciel.
    """
    rgb_f = rgb.astype(np.float32)
    normalized = (
        rgb_f[..., 0] + rgb_f[..., 1] * 256.0 + rgb_f[..., 2] * (256.0 * 256.0)
    ) / (256.0**3 - 1.0)
    meters = normalized * 1000.0
    return np.clip(meters, 0.0, max_depth_m).astype(np.float32)


def decode_semantic_carla(rgb: np.ndarray) -> np.ndarray:
    """Retourne un ``np.uint8 (H, W)`` avec les class_ids CityScape (canal R)."""
    return rgb[..., 0].copy()


CITYSCAPE_PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [110, 190, 160],
        [170, 120, 50],
        [55, 90, 80],
        [45, 60, 150],
        [157, 234, 50],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
    ],
    dtype=np.uint8,
)


def colorize_semantic(class_ids: np.ndarray) -> np.ndarray:
    """class_ids (H, W) → RGB (H, W, 3) selon la palette CityScape."""
    safe = np.clip(class_ids, 0, len(CITYSCAPE_PALETTE) - 1)
    return CITYSCAPE_PALETTE[safe]


def pack_instance_carla(rgb: np.ndarray) -> np.ndarray:
    """Packe les 3 canaux en uint32 : ``(class_id << 16) | (G << 8) | B``.

    Unpack côté consommateur ::

        class_id    = (packed >> 16) & 0xFF
        instance_id = packed & 0xFFFF

    ``instance_id == 0`` = pixel non trackable (fond / classe non comptée).
    """
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


_GOLDEN_RATIO_CONJUGATE = 0.6180339887


def colorize_instance(packed: np.ndarray) -> np.ndarray:
    """Rend une map instance packée en RGB. instance_id=0 → noir, sinon
    couleur HSV golden-ratio déterministe (même teinte d'une frame à l'autre
    pour un même objet)."""
    instance_ids = (packed & 0xFFFF).astype(np.uint32)
    unique_ids = np.unique(instance_ids)

    rgb_lookup = np.zeros((len(unique_ids), 3), dtype=np.uint8)
    for i, iid in enumerate(unique_ids):
        if iid == 0:
            continue
        hue = (float(iid) * _GOLDEN_RATIO_CONJUGATE) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.95)
        rgb_lookup[i] = (int(r * 255), int(g * 255), int(b * 255))

    sort_idx = np.searchsorted(unique_ids, instance_ids)
    return rgb_lookup[sort_idx]
