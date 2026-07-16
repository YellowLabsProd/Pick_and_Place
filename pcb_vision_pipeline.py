"""
PCB Component Detection + Grasp-Point Estimation Pipeline
============================================================

A project combining modern deep-learning detection (YOLOv8) with classic computer vision
(OpenCV contour/pose estimation) to demonstrate a perception pipeline for robotic PCB disassembly / sorting.

Usage:
    # Train a detector on a Roboflow dataset
    python pcb_vision_pipeline.py train --api-key YOUR_KEY \
        --workspace WORKSPACE_NAME --project PROJECT_NAME --version 1 --epochs 50

    pcb_vision_pipeline.py train --api-key "WxTeKtZjLNxHz8097ZB7" --workspace "test-ewfic" \
    --project "pcb-components-cmx11" --version 1 --epochs 50

    # Run detection + grasp-point estimation on an image
    python pcb_vision_pipeline.py detect --weights runs/detect/pcb_components/weights/best.pt \
        --source path/to/image_or_folder --conf 0.4 --out out/

    pcb_vision_pipeline.py detect --weights runs/detect/pcb_components/weights/best.pt --source pcb-components-1/valid \
    --conf 0.4 --out out/

    # Calibrate a camera from checkerboard photos
    python pcb_vision_pipeline.py calibrate --images calib_images/ \
        --pattern 9x6 --out calibration.yaml

Requirements:
    pip install ultralytics opencv-python roboflow pyyaml numpy
"""

import argparse
import glob
import os
import sys
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dataset download + YOLOv8 fine-tuning
# ---------------------------------------------------------------------------

def download_dataset(api_key: str, workspace: str, project: str, version: int, fmt: str = "yolov8"):
    """Download a labeled dataset from Roboflow Universe."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(fmt)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def train_detector(data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16, model_size: str = "yolov8n.pt",
                   run_name: str = "pcb_components"):
    """Fine-tune a YOLOv8 model on the downloaded PCB component dataset."""
    from ultralytics import YOLO

    model = YOLO(model_size)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
    )
    weights_path = os.path.join("runs", "detect", run_name, "weights", "best.pt")
    print(f"Training complete. Best weights: {weights_path}")
    return weights_path


def cmd_train(args):
    if args.data_yaml:
        data_yaml = args.data_yaml
    else:
        if not (args.api_key and args.workspace and args.project):
            sys.exit("Provide either --data-yaml, or --api-key/--workspace/--project to download a dataset.")
        location = download_dataset(args.api_key, args.workspace, args.project, args.version)
        data_yaml = os.path.join(location, "data.yaml")

    train_detector(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model_size,
        run_name=args.run_name,
    )


# ---------------------------------------------------------------------------
# Grasp-point / orientation estimation (classic CV)
# ---------------------------------------------------------------------------

def estimate_grasp_point(image: np.ndarray, box: tuple):
    """
    Given a full image and a bounding box (x1, y1, x2, y2), find the
    grasp point (centroid) and approach angle of the largest contour
    inside that box using thresholding + minAreaRect.

    Returns ((grasp_x, grasp_y), angle_degrees) or None if no contour found.
    """
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(largest)
    grasp_point = (int(x1 + cx), int(y1 + cy))
    return grasp_point, angle


def annotate_image(image: np.ndarray, boxes, classes, confs, class_names):
    """Draw YOLO boxes/labels plus grasp points/angles on the image."""
    annotated = image.copy()

    for box, cls_id, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls_id)]} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        grasp = estimate_grasp_point(image, box)
        if grasp:
            (gx, gy), angle = grasp
            cv2.circle(annotated, (gx, gy), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f"{angle:.0f} deg", (gx + 8, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return annotated


def run_detection(weights: str, source: str, conf: float, out_dir: str):
    """Run YOLOv8 inference on an image or folder, then overlay grasp points."""
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(weights)

    if os.path.isdir(source):
        image_paths = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in glob.glob(os.path.join(source, ext))
        )
    else:
        image_paths = [source]

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Skipping unreadable file: {path}")
            continue

        results = model.predict(source=path, conf=conf, verbose=False)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else []
        classes = r.boxes.cls.cpu().numpy() if len(r.boxes) else []
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else []

        annotated = annotate_image(image, boxes, classes, confs, r.names)

        out_path = os.path.join(out_dir, os.path.basename(path))
        cv2.imwrite(out_path, annotated)
        print(f"{os.path.basename(path)}: {len(boxes)} component(s) detected -> {out_path}")


def cmd_detect(args):
    run_detection(args.weights, args.source, args.conf, args.out)


# ---------------------------------------------------------------------------
# Day 2 — Camera calibration
# ---------------------------------------------------------------------------

def calibrate_camera(image_dir: str, pattern_size: tuple, out_path: str):
    """
    Calibrate a camera from checkerboard photos.
    pattern_size = (inner_corners_x, inner_corners_y), e.g. (9, 6)
    """
    import yaml

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    gray_shape = None

    image_files = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(image_dir, ext))
    )
    if not image_files:
        sys.exit(f"No calibration images found in {image_dir}")

    used = 0
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            used += 1

    if used < 5:
        print(f"Warning: only {used} usable calibration images found. "
              f"10-15+ is recommended for a stable result.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    print(f"Calibration RMS reprojection error: {ret:.4f}")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    with open(out_path, "w") as f:
        yaml.safe_dump({
            "rms_error": float(ret),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.ravel().tolist(),
            "image_width": gray_shape[0],
            "image_height": gray_shape[1],
            "images_used": used,
        }, f)
    print(f"Saved calibration to {out_path}")


def cmd_calibrate(args):
    x, y = (int(v) for v in args.pattern.lower().split("x"))
    calibrate_camera(args.images, (x, y), args.out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="PCB component vision pipeline (Day 1 + Day 2).")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    # p_train = sub.add_parser("train", help="Day 1: download dataset (optional) and fine-tune YOLOv8.")
    p_train = sub.add_parser("train", help="Download dataset (optional) and fine-tune YOLOv8.")
    p_train.add_argument("--data-yaml", help="Path to an already-downloaded data.yaml (skips download).")
    p_train.add_argument("--api-key", help="Roboflow API key.")
    p_train.add_argument("--workspace", help="Roboflow workspace name.")
    p_train.add_argument("--project", help="Roboflow project name.")
    p_train.add_argument("--version", type=int, default=1, help="Roboflow dataset version.")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--model-size", default="yolov8n.pt",
                         help="yolov8n.pt (fast) / yolov8s.pt (more accurate).")
    p_train.add_argument("--run-name", default="pcb_components")
    p_train.set_defaults(func=cmd_train)

    # detect
    # p_detect = sub.add_parser("detect", help="Day 2: run detection + grasp-point estimation.")
    p_detect = sub.add_parser("detect", help="Run detection + grasp-point estimation.")
    p_detect.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    p_detect.add_argument("--source", required=True, help="Image file or folder of images.")
    p_detect.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    p_detect.add_argument("--out", default="out", help="Output folder for annotated images.")
    p_detect.set_defaults(func=cmd_detect)

    # calibrate
    # p_calib = sub.add_parser("calibrate", help="Day 2: camera calibration from checkerboard photos.")
    p_calib = sub.add_parser("calibrate", help="Camera calibration from checkerboard photos.")
    p_calib.add_argument("--images", required=True, help="Folder of checkerboard photos.")
    p_calib.add_argument("--pattern", default="9x6", help="Inner corners as WxH, e.g. 9x6.")
    p_calib.add_argument("--out", default="calibration.yaml")
    p_calib.set_defaults(func=cmd_calibrate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
