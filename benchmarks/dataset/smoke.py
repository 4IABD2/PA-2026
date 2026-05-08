"""Smoke tests for src/dataset/ — verify contracts without CARLA."""

from __future__ import annotations


def test_high_level_command_enum_values():
    """HighLevelCommand exposes the 4 values documented in the root README."""
    from src.dataset.command_planner import HighLevelCommand

    assert HighLevelCommand.LEFT == "left"
    assert HighLevelCommand.RIGHT == "right"
    assert HighLevelCommand.STRAIGHT == "straight"
    assert HighLevelCommand.LANE_FOLLOW == "lane_follow"

    # All values are strings (StrEnum-like)
    for member in HighLevelCommand:
        assert isinstance(member.value, str)


import json
from pathlib import Path

import pandas as pd
import pytest


def test_manifest_csv_columns(tmp_path: Path):
    """ManifestWriter.flush_csv produces the 11 ordered columns from the root README."""
    from src.dataset.command_planner import HighLevelCommand
    from src.dataset.expert_driver import ExpertControls
    from src.dataset.manifest_writer import ManifestWriter

    writer = ManifestWriter(tmp_path, town="Town01", weather="ClearNoon")
    writer.append_row(
        frame_id=0,
        image_path="images/000000.jpg",
        timestamp=0.0,
        command=HighLevelCommand.STRAIGHT,
        expert=ExpertControls(steer=0.0, throttle=0.5, brake=0.0, speed_kmh=0.0),
        is_collision=False,
    )
    writer.append_row(
        frame_id=1,
        image_path="images/000001.jpg",
        timestamp=2.0,
        command=HighLevelCommand.STRAIGHT,
        expert=ExpertControls(steer=-0.02, throttle=0.5, brake=0.0, speed_kmh=12.5),
        is_collision=False,
    )
    writer.flush_csv()

    csv_path = tmp_path / "manifest.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)

    expected_cols = [
        "frame_id",
        "image_path",
        "timestamp",
        "command",
        "speed_kmh",
        "steer",
        "throttle",
        "brake",
        "is_collision",
        "town",
        "weather",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2
    assert df.iloc[0]["command"] == "straight"
    assert df.iloc[1]["speed_kmh"] == 12.5


def test_metadata_json_schema(tmp_path: Path):
    """ManifestWriter.write_metadata produces a JSON with the minimal required fields."""
    from src.dataset.manifest_writer import ManifestWriter

    writer = ManifestWriter(tmp_path, town="Town01", weather="ClearNoon")
    writer.write_metadata(
        run_id="2026-05-08_town01_clearnoon",
        carla_version="0.9.16",
        fps=20,
        capture_every_n_ticks=40,
        n_frames=10,
        n_npc_vehicles=5,
        n_npc_walkers=5,
        duration_sec_target=20,
        duration_sec_actual=19.85,
        camera_pov={"location": [0.30, 0.0, 1.50], "rotation": [0, 0, -5]},
        image_resolution=[1280, 720],
        fov=90,
        seed=42,
    )

    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.exists()
    data = json.loads(metadata_path.read_text())

    required_fields = {
        "run_id",
        "town",
        "weather",
        "carla_version",
        "fps",
        "capture_every_n_ticks",
        "n_frames",
        "camera_pov",
    }
    assert required_fields.issubset(data.keys())
    assert data["carla_version"] == "0.9.16"
    assert data["town"] == "Town01"


def test_yolo_label_format(tmp_path: Path):
    """YoloLabeler.save writes a file in '<class> <x_c> <y_c> <w> <h>' normalized [0,1] format."""
    from src.dataset.yolo_labels import YOLO_CLASS_MAPPING, YoloLabeler

    # Mapping has the 3 documented classes
    assert YOLO_CLASS_MAPPING == {"vehicle": 0, "walker": 1, "traffic_light": 2}

    # Test only the save() method with given input labels
    # (compute_labels() raises NotImplementedError in the skeleton)
    label_path = tmp_path / "000000.txt"
    fake_labels = [
        (0, 0.5, 0.5, 0.2, 0.3),  # vehicle at center
        (1, 0.1, 0.9, 0.05, 0.1),  # walker at bottom-left
    ]
    YoloLabeler.write_labels_to_file(fake_labels, label_path)

    content = label_path.read_text().strip().splitlines()
    assert len(content) == 2
    assert content[0] == "0 0.500000 0.500000 0.200000 0.300000"
    parts = content[1].split()
    assert parts[0] == "1"
    for v in parts[1:]:
        f = float(v)
        assert 0.0 <= f <= 1.0


import numpy as np


def test_depth_decoding_carla_format():
    """decode_carla_depth(R, G, B) returns depth in meters following the CARLA formula."""
    from src.dataset.depth_capture import decode_carla_depth

    # Pixel (R=0, G=0, B=0) -> 0 m (very close)
    depth = decode_carla_depth(np.array([[[0, 0, 0]]], dtype=np.uint8))
    assert depth.shape == (1, 1)
    assert depth[0, 0] == 0.0

    # Pixel (R=255, G=255, B=255) -> 1000 m (max), clipped to 100 by default
    depth = decode_carla_depth(
        np.array([[[255, 255, 255]]], dtype=np.uint8), max_depth_m=100.0
    )
    assert depth[0, 0] == 100.0

    # Arbitrary pixel (1, 0, 0) -> 1 / (256^3-1) * 1000 m ~= 0.0596 micrometers
    depth = decode_carla_depth(np.array([[[1, 0, 0]]], dtype=np.uint8))
    assert 0.0 < depth[0, 0] < 0.001  # micrometer order

    # dtype = float32
    assert depth.dtype == np.float32


def test_collector_init_refuses_existing_non_empty_dir(tmp_path: Path):
    """DatasetCollector raises ValueError if output_dir already contains files."""
    from src.dataset.collector import DatasetCollector

    # Create a file in tmp_path -> non empty
    (tmp_path / "manifest.csv").write_text("dummy")

    with pytest.raises(ValueError, match="not empty"):
        DatasetCollector(
            output_dir=tmp_path,
            duration_sec=1,
            host="dummy-no-connect",  # not reached, we test early init
        )


def test_collector_init_accepts_empty_dir(tmp_path: Path):
    """DatasetCollector accepts an empty or non-existing output_dir."""
    from src.dataset.collector import DatasetCollector

    new_dir = tmp_path / "fresh_run"
    # Must instantiate without a CARLA connection (validation only, no connect)
    collector = DatasetCollector(
        output_dir=new_dir,
        duration_sec=1,
        host="dummy-no-connect",
    )
    assert collector.output_dir == new_dir


def test_semantic_decode_carla_format():
    """decode_semantic_carla extracts class IDs from CARLA's R channel."""
    from src.dataset.semantic_capture import decode_semantic_carla

    # CARLA encodes class_id in R channel only. G and B are 0 for semantic.
    # Build a 2x2 image with class IDs 0, 1 (Roads), 14 (Car), 24 (RoadLine).
    rgb = np.array(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[14, 0, 0], [24, 0, 0]],
        ],
        dtype=np.uint8,
    )
    class_ids = decode_semantic_carla(rgb)

    assert class_ids.shape == (2, 2)
    assert class_ids.dtype == np.uint8
    assert class_ids[0, 0] == 0
    assert class_ids[0, 1] == 1
    assert class_ids[1, 0] == 14
    assert class_ids[1, 1] == 24


def test_semantic_palette_colorize():
    """colorize_semantic returns RGB with documented CARLA 0.9.13+ CityScape colors per class."""
    from src.dataset.semantic_capture import CITYSCAPE_PALETTE, colorize_semantic

    # Palette has 29 entries (class IDs 0-28, CARLA 0.9.13+ mapping), each (3,) uint8.
    assert CITYSCAPE_PALETTE.shape == (29, 3)
    assert CITYSCAPE_PALETTE.dtype == np.uint8

    # Class 0 (Unlabeled) is black.
    assert tuple(CITYSCAPE_PALETTE[0]) == (0, 0, 0)
    # Class 1 (Roads) is the documented (128, 64, 128).
    assert tuple(CITYSCAPE_PALETTE[1]) == (128, 64, 128)
    # Class 14 (Car) is the documented (0, 0, 142).
    assert tuple(CITYSCAPE_PALETTE[14]) == (0, 0, 142)
    # Class 24 (RoadLine) is the documented (157, 234, 50).
    assert tuple(CITYSCAPE_PALETTE[24]) == (157, 234, 50)

    # Colorize a 1x3 frame (Unlabeled, Roads, Car).
    class_ids = np.array([[0, 1, 14]], dtype=np.uint8)
    viz = colorize_semantic(class_ids)
    assert viz.shape == (1, 3, 3)
    assert viz.dtype == np.uint8
    assert tuple(viz[0, 0]) == (0, 0, 0)
    assert tuple(viz[0, 1]) == (128, 64, 128)
    assert tuple(viz[0, 2]) == (0, 0, 142)


def test_instance_packing_format():
    """pack_instance_carla packs (R, G, B) into uint32 (class_id<<16|G<<8|B)."""
    from src.dataset.instance_capture import pack_instance_carla

    # CARLA encodes:
    #   R = semantic class ID (e.g. 10 = Vehicles)
    #   G+B = instance_id high+low byte (G=0x12, B=0x34 → instance_id=0x1234)
    rgb = np.array([[[10, 0x12, 0x34]]], dtype=np.uint8)
    packed = pack_instance_carla(rgb)

    assert packed.shape == (1, 1)
    assert packed.dtype == np.uint32

    expected = (10 << 16) | (0x12 << 8) | 0x34  # 0x0A1234
    assert packed[0, 0] == expected

    # Unpack: class_id = (packed >> 16) & 0xFF, instance_id = packed & 0xFFFF.
    assert ((packed[0, 0] >> 16) & 0xFF) == 10
    assert (packed[0, 0] & 0xFFFF) == 0x1234

    # Edge case: all zeros (background pixel).
    rgb_zero = np.array([[[0, 0, 0]]], dtype=np.uint8)
    assert pack_instance_carla(rgb_zero)[0, 0] == 0


def test_instance_viz_color_is_deterministic():
    """colorize_instance returns the same RGB for the same instance ID across frames."""
    from src.dataset.instance_capture import (
        colorize_instance,
        pack_instance_carla,
    )

    # Two distinct frames, both with one pixel of instance_id=0x1234, class=10.
    frame1 = np.array([[[10, 0x12, 0x34]]], dtype=np.uint8)
    frame2 = np.array([[[10, 0x12, 0x34]]], dtype=np.uint8)
    packed1 = pack_instance_carla(frame1)
    packed2 = pack_instance_carla(frame2)

    viz1 = colorize_instance(packed1)
    viz2 = colorize_instance(packed2)

    assert viz1.shape == (1, 1, 3)
    assert viz1.dtype == np.uint8
    # Same instance_id → identical RGB (determinism).
    assert tuple(viz1[0, 0]) == tuple(viz2[0, 0])
    # And it is not black (instance_id != 0).
    assert tuple(viz1[0, 0]) != (0, 0, 0)

    # instance_id == 0 → background → black.
    bg = np.zeros((1, 1), dtype=np.uint32)
    viz_bg = colorize_instance(bg)
    assert tuple(viz_bg[0, 0]) == (0, 0, 0)

    # Different instance IDs → different colors (with very high probability).
    p1 = pack_instance_carla(np.array([[[10, 0x00, 0x01]]], dtype=np.uint8))
    p2 = pack_instance_carla(np.array([[[10, 0x00, 0x02]]], dtype=np.uint8))
    assert tuple(colorize_instance(p1)[0, 0]) != tuple(colorize_instance(p2)[0, 0])


def test_collector_output_subdirs_includes_segmentation():
    """OUTPUT_SUBDIRS lists the 7 expected dataset subdirectories (no duplicates)."""
    from src.dataset.collector import OUTPUT_SUBDIRS

    expected = {
        "images",
        "depth",
        "labels_yolo",
        "semantic",
        "semantic_viz",
        "instance",
        "instance_viz",
    }
    assert set(OUTPUT_SUBDIRS) == expected
    assert len(OUTPUT_SUBDIRS) == len(expected), "OUTPUT_SUBDIRS contains duplicates"
