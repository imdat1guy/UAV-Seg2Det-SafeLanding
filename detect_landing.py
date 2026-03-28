"""
Quick inference script for oriented safe-landing-zone detection.

Usage:
    python detect_landing.py path/to/image.jpg
    python detect_landing.py path/to/image.jpg --weights path/to/best.pt
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ── class names (must match training order) ──────────────────────────
CLASS_NAMES = ["L2-safe", "L3-safe"]


# ── geometry helpers ─────────────────────────────────────────────────
def order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def rasterize_quad_norm(pts_norm: np.ndarray, W: int, H: int) -> np.ndarray:
    pts = pts_norm.copy()
    pts[:, 0] = np.clip(pts[:, 0] * W, 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1] * H, 0, H - 1)
    pts = order_quad_clockwise(pts).astype(np.float32)
    poly = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(poly, pts.astype(np.int32), 1)
    return poly


def draw_obb_list(img_bgr, dets, class_colors=None, thickness=18):
    if class_colors is None:
        class_colors = {0: (255, 0, 0), 1: (5, 220, 5)}
    H, W = img_bgr.shape[:2]
    for d in reversed(dets):
        cls = d["cls"]
        pts = d["pts_norm"].copy()
        pts[:, 0] = np.clip(pts[:, 0] * W, 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1] * H, 0, H - 1)
        pts = order_quad_clockwise(pts).astype(np.int32)
        color = class_colors.get(cls, (0, 255, 255))
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness)
    return img_bgr

def select_best_box(dets, W, H):
    selected = None
    best_key = None
    for d in dets:
        cls = d["cls"]
        if cls not in (0, 1):
            continue
        mask = rasterize_quad_norm(d["pts_norm"], W, H)
        area = int(mask.sum())
        if area <= 0:
            continue
        level_rank = 1 if cls == 1 else 0  # L3 > L2
        key = (level_rank, area, d["conf"])
        if best_key is None or key > best_key:
            best_key = key
            selected = d
    return selected


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detect safe landing zones in a drone image."
    )
    parser.add_argument("image", type=str, help="Path to the input image.")
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained/weights/best.pt",
        help="Path to YOLO-OBB weights (default: pretrained/weights/best.pt).",
    )
    parser.add_argument("--imgsz", type=int, default=1920, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    img_path = Path(args.image)
    assert img_path.exists(), f"Image not found: {img_path}"

    weights = Path(args.weights)
    assert weights.exists(), f"Weights not found: {weights}"

    # ── load model & run inference ───────────────────────────────────
    model = YOLO(str(weights))
    results = model.predict(
        source=str(img_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.50,
        max_det=100,
        verbose=False,
    )
    result = results[0]

    # ── extract detections from YOLO OBB result ─────────────────────
    dets = []
    if result.obb is not None and len(result.obb):
        obb = result.obb
        for i in range(len(obb)):
            cls = int(obb.cls[i].item())
            conf = float(obb.conf[i].item())
            pts_norm = obb.xyxyxyxyn[i].cpu().numpy().reshape(4, 2)
            dets.append({"cls": cls, "conf": conf, "pts_norm": pts_norm})

    # ── read original image ──────────────────────────────────────────
    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"Could not read image: {img_path}"
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── build figure: original | all detections | selected landing ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # col 0: original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # col 1: all detected OBBs
    det_img = img_bgr.copy()
    draw_obb_list(det_img, dets)
    axes[1].imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Detected OBBs ({len(dets)})")
    axes[1].axis("off")

    # col 2: selected landing box
    sel_img = img_bgr.copy()
    best = select_best_box(dets, W, H)
    if best is not None:
        draw_obb_list(sel_img, [best])
        cls_name = CLASS_NAMES[best["cls"]] if best["cls"] < len(CLASS_NAMES) else f"cls{best['cls']}"
        title = f"Selected Landing Zone ({cls_name}, conf={best['conf']:.2f})"
    else:
        title = "No Valid Landing Zone"
        cv2.putText(
            sel_img, "No valid selection", (40, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
        )
    axes[2].imshow(cv2.cvtColor(sel_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
