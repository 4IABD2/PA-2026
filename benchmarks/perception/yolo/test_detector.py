"""Unit tests for YoloDetector.detect() -- no real YOLO model, no GPU."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from src.perception.yolo.detector import YoloDetector


def _make_box(cls_id: int, xyxy: tuple[int, int, int, int], conf: float = 0.9) -> Mock:
    box = Mock()
    box.cls = [Mock(cpu=Mock(return_value=cls_id))]
    box.conf = [Mock(cpu=Mock(return_value=conf))]
    xyxy_tensor = Mock()
    xyxy_tensor.cpu.return_value.int.return_value.tolist.return_value = list(xyxy)
    box.xyxy = [xyxy_tensor]
    return box


def _make_detector(boxes: list) -> YoloDetector:
    detector = YoloDetector.__new__(YoloDetector)
    results = [Mock(boxes=boxes)]
    detector.model = Mock(return_value=results)
    return detector


def test_detect_skips_degenerate_light_bbox_without_crashing():
    """A zero-width traffic-light box must be skipped, not crash cv2.cvtColor."""
    degenerate_light = _make_box(cls_id=2, xyxy=(50, 50, 50, 80))  # x1 == x2
    detector = _make_detector([degenerate_light])
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert detections == []


def test_detect_skips_zero_height_light_bbox():
    """A zero-height traffic-light box must also be skipped."""
    degenerate_light = _make_box(cls_id=3, xyxy=(10, 40, 30, 40))  # y1 == y2
    detector = _make_detector([degenerate_light])
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert detections == []


def test_detect_keeps_non_light_detection_with_degenerate_bbox():
    """A degenerate box on a NON-light class must still be reported (only the
    light-crop path can crash; other classes never crop the image)."""
    degenerate_vehicle = _make_box(cls_id=0, xyxy=(50, 50, 50, 50))
    detector = _make_detector([degenerate_vehicle])
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert len(detections) == 1
    assert detections[0].bbox == (50, 50, 50, 50)


def test_detect_classifies_valid_light_bbox(monkeypatch):
    """A normal, non-degenerate light bbox must still go through classify_tl_color."""
    import src.perception.yolo.detector as detector_module

    monkeypatch.setattr(detector_module, "classify_tl_color", lambda crop: (2, "red"))
    valid_light = _make_box(cls_id=2, xyxy=(10, 10, 30, 30))
    detector = _make_detector([valid_light])
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert len(detections) == 1
    assert detections[0].bbox == (10, 10, 30, 30)
