"""Unit tests for PerceptionPipeline's calibration handling -- YoloDetector and
DepthEstimator are mocked so no real model ever loads."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.perception.pipeline import PerceptionPipeline

_MISSING_CALIB = "/definitely/does/not/exist.json"


def test_require_calibration_true_raises_before_loading_models():
    """A missing calibration path must raise immediately, before either heavy
    model is constructed."""
    with (
        patch("src.perception.pipeline.YoloDetector") as mock_yolo,
        patch("src.perception.pipeline.DepthEstimator") as mock_depth,
    ):
        with pytest.raises(RuntimeError, match=_MISSING_CALIB.replace(".", r"\.")):
            PerceptionPipeline(
                require_calibration=True,
                depth_calibration=_MISSING_CALIB,
            )

    mock_yolo.assert_not_called()
    mock_depth.assert_not_called()


def test_require_calibration_false_warns_and_proceeds(capsys):
    """Default behaviour: missing calibration only warns and still constructs
    both models (existing non-training callers must keep working)."""
    with (
        patch("src.perception.pipeline.YoloDetector") as mock_yolo,
        patch("src.perception.pipeline.DepthEstimator") as mock_depth,
    ):
        PerceptionPipeline(
            require_calibration=False,
            depth_calibration=_MISSING_CALIB,
        )

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "uncalibrated" in captured.out.lower() or "not found" in captured.out.lower()
    mock_yolo.assert_called_once()
    mock_depth.assert_called_once()


@pytest.mark.parametrize("require_calibration", [True, False])
def test_existing_calibration_path_never_warns_or_raises(
    tmp_path, capsys, require_calibration
):
    """A calibration file that DOES exist must never raise nor warn, whether
    or not calibration is required."""
    calib_file = tmp_path / "calibration.json"
    calib_file.write_text("{}")

    with (
        patch("src.perception.pipeline.YoloDetector") as mock_yolo,
        patch("src.perception.pipeline.DepthEstimator") as mock_depth,
    ):
        PerceptionPipeline(
            require_calibration=require_calibration,
            depth_calibration=str(calib_file),
        )

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
    mock_yolo.assert_called_once()
    mock_depth.assert_called_once()
