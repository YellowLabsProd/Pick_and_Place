"""
Tests for pcb_vision_pipeline.py

Run with:
    pytest tests/ -v

These tests avoid requiring a GPU, real trained weights, or a live Roboflow
account by:
  - exercising the pure OpenCV/argparse logic directly (grasp-point
    estimation, annotation, CLI parsing), and
  - stubbing out `ultralytics` and `roboflow` in sys.modules for the
    functions that only orchestrate calls into those libraries.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import cv2

import pcb_vision_pipeline as pvp


# ---------------------------------------------------------------------------
# estimate_grasp_point
# ---------------------------------------------------------------------------

def make_square_image(size=200, square=(60, 60, 140, 140), color=0, bg=255):
    """White image with a filled black square, for predictable contours."""
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    x1, y1, x2, y2 = square
    img[y1:y2, x1:x2] = color
    return img


def test_estimate_grasp_point_returns_center_of_square():
    img = make_square_image()
    box = (40, 40, 160, 160)  # bounding box that fully contains the square
    result = pvp.estimate_grasp_point(img, box)

    assert result is not None
    (gx, gy), angle = result

    # centroid of a centered square should land near the square's center (100, 100)
    assert abs(gx - 100) <= 5
    assert abs(gy - 100) <= 5
    assert isinstance(angle, float)


def test_estimate_grasp_point_empty_crop_returns_none():
    img = make_square_image()
    # zero-area box (x1 == x2) -> crop.size == 0
    result = pvp.estimate_grasp_point(img, (50, 50, 50, 120))
    assert result is None


def test_estimate_grasp_point_no_contour_returns_none_or_point():
    # A perfectly uniform crop still produces a thresholded blob via OTSU in
    # practice (noise-free uniform images can behave oddly with OTSU), so we
    # just assert the function doesn't raise and returns a valid shape.
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    result = pvp.estimate_grasp_point(img, (10, 10, 190, 190))
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


# ---------------------------------------------------------------------------
# annotate_image
# ---------------------------------------------------------------------------

def test_annotate_image_no_boxes_returns_same_shape():
    img = make_square_image()
    annotated = pvp.annotate_image(img, [], [], [], class_names={0: "component"})
    assert annotated.shape == img.shape
    # with no boxes drawn, image should be identical to the input
    assert np.array_equal(annotated, img)


def test_annotate_image_draws_box_and_grasp_point():
    img = make_square_image()
    boxes = [(40, 40, 160, 160)]
    classes = [0]
    confs = [0.87]
    class_names = {0: "resistor"}

    annotated = pvp.annotate_image(img, boxes, classes, confs, class_names)

    assert annotated.shape == img.shape
    # the annotated image should differ from the original (box/label/point drawn)
    assert not np.array_equal(annotated, img)


# ---------------------------------------------------------------------------
# build_parser (CLI)
# ---------------------------------------------------------------------------

def test_parser_train_defaults():
    parser = pvp.build_parser()
    args = parser.parse_args(["train", "--data-yaml", "data.yaml"])
    assert args.command == "train"
    assert args.data_yaml == "data.yaml"
    assert args.epochs == 50
    assert args.imgsz == 640
    assert args.batch == 16
    assert args.model_size == "yolov8n.pt"
    assert args.run_name == "pcb_components"


def test_parser_detect_requires_weights_and_source():
    parser = pvp.build_parser()
    args = parser.parse_args([
        "detect", "--weights", "best.pt", "--source", "images/"
    ])
    assert args.command == "detect"
    assert args.weights == "best.pt"
    assert args.source == "images/"
    assert args.conf == 0.4
    assert args.out == "out"


def test_parser_detect_missing_required_arg_raises():
    parser = pvp.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["detect", "--source", "images/"])  # missing --weights


def test_parser_calibrate_defaults():
    parser = pvp.build_parser()
    args = parser.parse_args(["calibrate", "--images", "calib/"])
    assert args.command == "calibrate"
    assert args.images == "calib/"
    assert args.pattern == "9x6"
    assert args.out == "calibration.yaml"


# ---------------------------------------------------------------------------
# train_detector / run_detection — mock ultralytics so no real model runs
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ultralytics(monkeypatch):
    """Stub out `ultralytics` so `from ultralytics import YOLO` succeeds
    without pulling in the real (heavy) dependency."""
    fake_module = types.ModuleType("ultralytics")
    mock_model_instance = MagicMock()
    mock_model_instance.train.return_value = MagicMock()

    mock_yolo_cls = MagicMock(return_value=mock_model_instance)
    fake_module.YOLO = mock_yolo_cls

    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    return mock_yolo_cls, mock_model_instance


def test_train_detector_calls_yolo_and_returns_weights_path(fake_ultralytics, tmp_path):
    mock_yolo_cls, mock_model_instance = fake_ultralytics

    weights_path = pvp.train_detector(
        data_yaml="data.yaml",
        epochs=5,
        imgsz=320,
        batch=8,
        model_size="yolov8n.pt",
        run_name="test_run",
    )

    mock_yolo_cls.assert_called_once_with("yolov8n.pt")
    mock_model_instance.train.assert_called_once_with(
        data="data.yaml", epochs=5, imgsz=320, batch=8, name="test_run"
    )
    assert weights_path == pvp.os.path.join(
        "runs", "detect", "test_run", "weights", "best.pt"
    )


@pytest.fixture
def fake_roboflow(monkeypatch):
    fake_module = types.ModuleType("roboflow")
    mock_dataset = MagicMock()
    mock_dataset.location = "/tmp/fake_dataset"

    mock_version = MagicMock()
    mock_version.download.return_value = mock_dataset

    mock_project = MagicMock()
    mock_project.version.return_value = mock_version

    mock_workspace = MagicMock()
    mock_workspace.project.return_value = mock_project

    mock_rf_instance = MagicMock()
    mock_rf_instance.workspace.return_value = mock_workspace

    fake_module.Roboflow = MagicMock(return_value=mock_rf_instance)
    monkeypatch.setitem(sys.modules, "roboflow", fake_module)
    return fake_module


def test_download_dataset_returns_location(fake_roboflow):
    location = pvp.download_dataset(
        api_key="fake_key", workspace="ws", project="proj", version=1
    )
    assert location == "/tmp/fake_dataset"


def test_run_detection_writes_annotated_images(fake_ultralytics, tmp_path, monkeypatch):
    mock_yolo_cls, mock_model_instance = fake_ultralytics

    # Build a fake YOLO result: one detected box
    fake_box = MagicMock()
    fake_box.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
    fake_box.cls.cpu.return_value.numpy.return_value = np.array([0])
    fake_box.conf.cpu.return_value.numpy.return_value = np.array([0.9])
    fake_box.__len__.return_value = 1

    fake_result = MagicMock()
    fake_result.boxes = fake_box
    fake_result.names = {0: "component"}

    mock_model_instance.predict.return_value = [fake_result]

    # Write a real temp image so cv2.imread succeeds
    img_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(img_path), make_square_image())

    out_dir = tmp_path / "out"
    pvp.run_detection(
        weights="fake_weights.pt",
        source=str(img_path),
        conf=0.4,
        out_dir=str(out_dir),
    )

    out_file = out_dir / "sample.jpg"
    assert out_file.exists()
