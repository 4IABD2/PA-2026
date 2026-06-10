"""Helpers numpy purs pour décoder les capteurs caméra CARLA.

Aucun import carla : ces fonctions opèrent sur des ``np.ndarray`` déjà extraits
du `raw_data` du sensor (cf. ``sensors.py``). On peut donc les utiliser hors
collecte (tests, post-process, notebooks d'analyse).

Contient aussi les constantes de POV caméra partagées par tous les capteurs.
"""

from __future__ import annotations

import colorsys

import numpy as np

# ----------------------------------------------------------------------------
# Caméra : POV partagé par tous les capteurs (rgb / depth / semantic / instance)
# ----------------------------------------------------------------------------
# Position hors body — les sensors semantic/instance ne respectent pas la
# transparence des matériaux, on les place donc au-dessus du toit Tesla
# (~1.50m, toit à ~1.44m) pour éviter de capturer l'intérieur du cockpit.
CAMERA_LOCATION = (0.30, 0.0, 1.50)
CAMERA_ROTATION_PITCH = -5.0

# ----------------------------------------------------------------------------
# Depth : décodage des 3 canaux RGB en distance (mètres)
# ----------------------------------------------------------------------------
# Formula CARLA : meters = ((R + G*256 + B*256**2) / (256**3 - 1)) * 1000.
# Plage physique 0-1000 m. On clippe à ``max_depth_m`` pour le ciel.


def decode_carla_depth(rgb: np.ndarray, max_depth_m: float = 1000.0) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    normalized = (
        rgb_f[..., 0] + rgb_f[..., 1] * 256.0 + rgb_f[..., 2] * (256.0 * 256.0)
    ) / (256.0**3 - 1.0)
    meters = normalized * 1000.0
    return np.clip(meters, 0.0, max_depth_m).astype(np.float32)


# ----------------------------------------------------------------------------
# Semantic : le canal R contient le class_id CityScape (0-28)
# ----------------------------------------------------------------------------


def decode_semantic_carla(rgb: np.ndarray) -> np.ndarray:
    """Retourne un ``np.uint8 (H, W)`` avec les class_ids CityScape."""
    return rgb[..., 0].copy()


# Palette CityScape officielle CARLA 0.9.13+ (29 classes, indices 0-28).
CITYSCAPE_PALETTE = np.array(
    [
        [0, 0, 0],  # 0  Unlabeled
        [128, 64, 128],  # 1  Roads
        [244, 35, 232],  # 2  SideWalks
        [70, 70, 70],  # 3  Building
        [102, 102, 156],  # 4  Wall
        [190, 153, 153],  # 5  Fence
        [153, 153, 153],  # 6  Pole
        [250, 170, 30],  # 7  TrafficLight
        [220, 220, 0],  # 8  TrafficSign
        [107, 142, 35],  # 9  Vegetation
        [152, 251, 152],  # 10 Terrain
        [70, 130, 180],  # 11 Sky
        [220, 20, 60],  # 12 Pedestrian
        [255, 0, 0],  # 13 Rider
        [0, 0, 142],  # 14 Car
        [0, 0, 70],  # 15 Truck
        [0, 60, 100],  # 16 Bus
        [0, 80, 100],  # 17 Train
        [0, 0, 230],  # 18 Motorcycle
        [119, 11, 32],  # 19 Bicycle
        [110, 190, 160],  # 20 Static
        [170, 120, 50],  # 21 Dynamic
        [55, 90, 80],  # 22 Other
        [45, 60, 150],  # 23 Water
        [157, 234, 50],  # 24 RoadLine
        [81, 0, 81],  # 25 Ground
        [150, 100, 100],  # 26 Bridge
        [230, 150, 140],  # 27 RailTrack
        [180, 165, 180],  # 28 GuardRail
    ],
    dtype=np.uint8,
)


def colorize_semantic(class_ids: np.ndarray) -> np.ndarray:
    """class_ids (H, W) → RGB (H, W, 3) selon la palette CityScape."""
    safe = np.clip(class_ids, 0, len(CITYSCAPE_PALETTE) - 1)
    return CITYSCAPE_PALETTE[safe]


# ----------------------------------------------------------------------------
# Instance : 3 canaux → uint32 packé (class_id << 16) | (G << 8) | B
# ----------------------------------------------------------------------------
# Unpack côté consommateur :
#     class_id    = (packed >> 16) & 0xFF
#     instance_id = packed & 0xFFFF
# instance_id == 0 = pixel non trackable (fond / classe non comptée).


def pack_instance_carla(rgb: np.ndarray) -> np.ndarray:
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
