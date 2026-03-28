#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeLanding Seg→Det OBB exporter (one-pass, optimized)

For each mask image:
  - iterate L3→L2 (→L1 if enabled), connected components
  - base OBB via minAreaRect on that component
  - contamination check (gap-wise tolerance)
  - if contaminated: salvage with three fallbacks
        1) refine_angle_scale_in_allowed_roi
        2) _max_inscribed_rect_in_allowed
        3) greedy_cover_with_rotated_maxrect_fast
  - write COCO + YOLOv8-OBB outputs

Requires: numpy, opencv-python, tqdm, matplotlib (if WRITE_VIZ)
"""

from __future__ import annotations
import json, math, csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm

# ------------------ CONFIG ------------------
DATA_ROOT = Path("../Data/training_set")
IM_DIR    = DATA_ROOT / "images"
LEV_DIR   = DATA_ROOT / "slz_out" / "masks_levels"

OUT_ROOT  = DATA_ROOT / "slz_out" / "det_obb"
YOLO_DIR  = OUT_ROOT / "greedy_labels_yolo_obb"     # per-image *.txt
COCO_JSON = OUT_ROOT / "greedy_slz_obb_all.json"    # one COCO-like file
VIZ_DIR   = OUT_ROOT / "diagnostics"         # optional visual overlays

# Safety thresholds / classes
MIN_LEVEL       = 2
INCLUDE_LEVEL1  = False
LEVEL_TO_CAT    = {1: "slz_l1", 2: "slz_l2", 3: "slz_l3"}

# Aircraft / scale (camera = Sony a6000 defaults)
IMG_W             = 6000
SENSOR_W_MM       = 23.5
ASSUMED_FOCAL_MM  = 16.0
PX_SIZE_MM        = SENSOR_W_MM / IMG_W  # mm/px (horizontal)
D_M               = 0.40   # metres, tip-to-tip rotor diagonal incl. guards

ALTS_CSV = DATA_ROOT / "slz_out" / "altitude" / "altitudes_final.csv"

# Contamination tolerance (gap-wise by level difference)
GAP_TOL_FRAC = {1: 0.10, 2: 0.03}  # ≤10% L-1, ≤3% L-2
GAP_TOL_MIN_PX = {1: 50, 2: 10}

# Salvage knobs
ROI_EDGE_PAD               = 1
GUARD_PX                   = 2
SALVAGE_MIN_FRAC_OF_COMP   = 0.70  # single-rect salvage must cover ≥70% of allowed
GREEDY_COVER_FRAC          = 0.80  # stop when residual covered to this fraction
GREEDY_MAX_RECTS           = 12
GREEDY_IOU_CAP             = 0.50
GREEDY_MAX_SIDE            = 1000  # downscale long side for speed

# Visualization
WRITE_VIZ = True

# -------------------------------------------

def gsd_from_altitude_m(alt_m: float) -> float:
    return alt_m * (PX_SIZE_MM / ASSUMED_FOCAL_MM)  # m/px

def load_GSDs(csv_path: Path) -> dict[str, float]:
    gsd = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            img_id = row["image_id"].strip()
            H = float(row["final_alt_m"])
            gsd[img_id] = gsd_from_altitude_m(H)
    return gsd

def amin_pixels(D_m: float, s_m_per_px: float, factor: float = 1.5) -> int:
    area_m2 = factor * (math.pi * (D_m/2.0)**2)
    return int(np.ceil(area_m2 / (s_m_per_px**2)))

def ensure_dirs():
    for p in [OUT_ROOT, YOLO_DIR, VIZ_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def load_levels_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    # Accept grayscale, BGR, BGRA
    if img.ndim == 2:
        lev = img
    else:
        # OpenCV channels: B,G,R,(A)
        lev = img[:,:,2]
    return lev.astype(np.uint8)

# -------------- geometry helpers --------------
def order_quad_clockwise(pts4: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts4, dtype=np.float32)

    # 1) centroid of the four points
    c = pts.mean(axis=0)

    # 2) angle of each point around the centroid
    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])

    # 3) sort by angle to get consistent circular order
    order = np.argsort(angles)
    ordered = pts[order]

    # 4) rotate so that index 0 is top-left (smallest y, then x)
    tl_idx = np.lexsort((ordered[:,0], ordered[:,1]))[0]
    ordered = np.roll(ordered, -tl_idx, axis=0)

    return ordered

def quad_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    A = boxA.astype(np.float32)
    B = boxB.astype(np.float32)
    inter_area, _ = cv2.intersectConvexConvex(A, B)
    if inter_area <= 0:
        return 0.0
    aA = abs(cv2.contourArea(A)); aB = abs(cv2.contourArea(B))
    return float(inter_area / (aA + aB - inter_area + 1e-6))

def rect_from_mask_component(comp_roi: np.ndarray) -> Tuple[float,float,float,float,float,np.ndarray]:
    """minAreaRect on a binary ROI component (uint8 0/1). Returns in ROI coords."""
    cnts, _ = cv2.findContours(comp_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    if w < h:
        w, h = h, w
        angle += 90.0
    angle = angle % 180.0
    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    box = order_quad_clockwise(box)
    return cx, cy, w, h, angle, box

def build_allowed_mask_for_component(lev_img: np.ndarray,
                                     lvl: int,
                                     roi_xywh: tuple[int,int,int,int],
                                     seed_comp: np.ndarray,
                                     connectivity: int = 8) -> np.ndarray:
    """L>=lvl AND connected (through ≥lvl) to the original component, inside ROI."""
    x, y, w, h = roi_xywh
    lev_roi = lev_img[y:y+h, x:x+w]
    allowed = (lev_roi >= lvl).astype(np.uint8)
    if allowed.sum() == 0:
        return np.zeros_like(seed_comp, dtype=np.uint8)

    num, labels = cv2.connectedComponents(allowed, connectivity)
    if num <= 1:
        return np.zeros_like(seed_comp, dtype=np.uint8)

    seed_labels = labels[seed_comp.astype(bool)]
    if seed_labels.size == 0:
        return seed_comp.copy()

    uniq, cnts = np.unique(seed_labels, return_counts=True)
    nz = uniq != 0
    if nz.sum() == 0:
        return seed_comp.copy()
    label = int(uniq[nz][np.argmax(cnts[nz])])
    return (labels == label).astype(np.uint8)

def zero_out_roi_frame(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask
    h, w = mask.shape[:2]
    p = min(pad, h//2, w//2)
    if p <= 0:
        return mask
    m = mask.copy()
    m[:p, :] = 0; m[-p:, :] = 0; m[:, :p] = 0; m[:, -p:] = 0
    return m

def _rect_inside_roi_ok(cx, cy, w, h, angle_deg,
                        allowed_roi: np.ndarray,
                        roi_xywh: tuple[int,int,int,int],
                        scratch: np.ndarray) -> tuple[bool, np.ndarray]:
    x, y, w_roi, h_roi = roi_xywh
    cx_r = cx - x; cy_r = cy - y
    box = cv2.boxPoints(((cx_r, cy_r), (w, h), angle_deg)).astype(np.float32)
    scratch[:] = 0
    cv2.fillConvexPoly(scratch, box.astype(np.int32), 1)
    ok = int((scratch & (1 - allowed_roi)).sum()) == 0
    return ok, box

def refine_angle_scale_in_allowed_roi(cx, cy, w, h, angle0,
                                      allowed_roi: np.ndarray,
                                      roi_xywh: tuple[int,int,int,int],
                                      ang_hint: Optional[float],
                                      ang_search: tuple[float,...] = (-45,-20,-10,0,+10,+20,+45),
                                      trans_px: int = 2,
                                      scale_iters: int = 12,
                                      early_stop_pct: float = 0.02):
    """Angle+scale refine in ROI coords; returns in *image* coords."""
    x, y, w_roi, h_roi = roi_xywh
    scratch = np.zeros_like(allowed_roi, dtype=np.uint8)
    # candidate angles
    cands = { (angle0 + a) % 180.0 for a in ang_search }
    cands.add(angle0 % 180.0)
    if ang_hint is not None:
        for a in ang_search:
            cands.add((ang_hint + a) % 180.0)
        cands.add(ang_hint % 180.0)
    tvals = [-trans_px, 0, trans_px] if trans_px > 0 else [0]

    def best_scale_at_angle(cx_i, cy_i, ang):
        lo, hi, best_box = 0.0, 1.0, None
        for _ in range(scale_iters):
            mid = 0.5*(lo+hi)
            ok, box_roi = _rect_inside_roi_ok(cx_i, cy_i, w*mid, h*mid, ang, allowed_roi, roi_xywh, scratch)
            if ok: best_box, lo = box_roi, mid
            else:  hi = mid
        if best_box is None:
            return None
        return lo, best_box

    best_area, best = 0.0, None
    consec_small_gain = 0
    for ang in sorted(cands, key=lambda a: abs(((a+90)%180)-90)):
        before = best_area
        local_best = None
        for dx in tvals:
            for dy in tvals:
                cx_i = float(np.clip(cx, x, x+w_roi-1)) + dx
                cy_i = float(np.clip(cy, y, y+h_roi-1)) + dy
                res = best_scale_at_angle(cx_i, cy_i, ang)
                if res is None: 
                    continue
                s, box_roi = res
                ww, hh = w*s, h*s
                area = ww*hh
                if (local_best is None) or (area > local_best[0]):
                    local_best = (area, cx_i, cy_i, ww, hh, ang, box_roi)
        if local_best and local_best[0] > best_area:
            best_area, best = local_best[0], local_best
        gain = 0.0 if before == 0 else (best_area - before) / max(1e-6, before)
        consec_small_gain = (consec_small_gain + 1) if (gain < early_stop_pct) else 0
        if consec_small_gain >= 3:
            break

    if best is None:
        return None
    _, cx_b, cy_b, wb, hb, ang_b, box_roi = best
    box_img = box_roi.copy(); box_img[:,0] += x; box_img[:,1] += y
    return cx_b, cy_b, wb, hb, ang_b, order_quad_clockwise(box_img)

def _max_inscribed_rect_in_allowed(allowed_mask: np.ndarray,
                                   angle_hint_deg: Optional[float],
                                   margin_px: int = 1,
                                   angle_probe: tuple[float, ...] = (+15, -15, +30, -30),
                                   iters: int = 25):
    """Largest inscribed rectangle fully inside allowed_mask; returns in mask coords."""
    m = (allowed_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    m_in = m if margin_px <= 0 else cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin_px+1, 2*margin_px+1)))
    if m_in.sum() == 0:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    (_, _), (w0, h0), ang0 = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if w0 < h0: w0, h0, ang0 = h0, w0, (ang0 + 90.0)
    ang0 = ang0 % 180.0
    angle_candidates = ([ang0] + [(ang0 + d) % 180.0 for d in angle_probe] + [0.0, 90.0]) \
                        if angle_hint_deg is None else [angle_hint_deg]

    dt = cv2.distanceTransform(m_in, cv2.DIST_L2, 5)
    ay, ax = np.unravel_index(np.argmax(dt), dt.shape)
    ax, ay = float(ax), float(ay)

    def _fits(w, h, ang_deg):
        box = cv2.boxPoints(((ax, ay), (w, h), ang_deg)).astype(np.float32)
        poly = np.zeros_like(m_in, dtype=np.uint8)
        cv2.fillConvexPoly(poly, box.astype(np.int32), 1)
        ok = int((poly & (1 - m_in)).sum()) == 0
        return ok, box

    best = None
    for ang in angle_candidates:
        lo, hi, ok_box = 0.0, 1.0, None
        for _ in range(iters):
            mid = 0.5*(lo+hi)
            ok, box = _fits(w0*mid, h0*mid, ang)
            if ok: ok_box, lo = box, mid
            else:  hi = mid
        if ok_box is None:
            continue
        area = (w0*lo)*(h0*lo)
        if (best is None) or (area > best[0]):
            best = (area, ang, w0*lo, h0*lo, ok_box)

    if best is None:
        return None
    _, ang_best, wb, hb, box = best
    return ax, ay, wb, hb, ang_best, order_quad_clockwise(box.copy())

# ---------- fast rotated max-rect greedy ----------
def largest_rectangle_in_binary_matrix(bin01: np.ndarray):
    """O(H*W) per image using 'largest rectangle in histogram' per row.
       Returns (area, top, left, height, width)."""
    H, W = bin01.shape
    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for r in range(H):
        row = bin01[r]
        heights = np.where(row > 0, heights + 1, 0)
        stack = []
        i = 0
        while i <= W:
            cur_h = heights[i] if i < W else 0
            if not stack or cur_h >= heights[stack[-1]]:
                stack.append(i); i += 1
            else:
                top = stack.pop()
                h = heights[top]
                left = stack[-1] + 1 if stack else 0
                w = i - left
                area = h * w
                if area > best[0]:
                    best = (area, r - h + 1, left, h, w)
    return best

def rotate_mask_keep_all(mask01: np.ndarray, angle_deg: float):
    """Rotate by angle around center with canvas expansion. Returns (rot01, M, Minv)."""
    H, W = mask01.shape[:2]
    center = (W * 0.5, H * 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)  # +angle = CCW
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nW = int(H * sin + W * cos)
    nH = int(H * cos + W * sin)
    M[0,2] += (nW * 0.5) - center[0]
    M[1,2] += (nH * 0.5) - center[1]
    rot = cv2.warpAffine(mask01.astype(np.uint8), M, (nW, nH),
                         flags=cv2.INTER_NEAREST, borderValue=0)
    Minv = cv2.invertAffineTransform(M)
    return (rot > 0).astype(np.uint8), M, Minv

def angle_candidates_from_mask(mask01: np.ndarray):
    """Small, strong set: 0, 90, and minAreaRect ±10."""
    cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [0.0, 90.0]
    (_, _), (w0,h0), ang0 = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if w0 < h0: ang0 += 90.0
    ang0 = ang0 % 180.0
    S = {0.0, 90.0, ang0, (ang0+10)%180.0, (ang0-10)%180.0}
    return sorted(S, key=lambda a: abs(((a+90)%180)-90))

def greedy_cover_with_rotated_maxrect_fast(allowed_mask: np.ndarray,
                                           guard_px: int = 2,
                                           cover_frac: float = 0.85,
                                           max_rects: int = 8,
                                           iou_cap: float = 0.5,
                                           angles: Optional[List[float]] = None,
                                           max_side: int = 1000):
    """Fast greedy: pre-rotate each angle once, largest AAR per view, update covered."""
    m = (allowed_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return []

    if guard_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*guard_px+1, 2*guard_px+1))
        m = cv2.erode(m, k)
        if m.sum() == 0:
            return []

    H, W = m.shape
    scale = 1.0
    ms = m
    if max(H, W) > max_side:
        scale = max_side / float(max(H, W))
        ms = cv2.resize(m, (int(round(W*scale)), int(round(H*scale))), interpolation=cv2.INTER_NEAREST)

    if angles is None:
        angles = angle_candidates_from_mask(ms)

    views = []
    for theta in angles:
        rot, M, Minv = rotate_mask_keep_all(ms, -theta)  # align theta-rects with axes
        views.append({"theta": float(theta), "rot": rot, "covered": np.zeros_like(rot, np.uint8),
                      "M": M, "Minv": Minv})

    placed_small = []
    covered_small = np.zeros_like(ms, np.uint8)
    total = int(ms.sum())

    for _ in range(max_rects):
        if covered_small.sum() / max(1, total) >= cover_frac:
            break

        best = None  # (area, vi, (t,l,h,w))
        for vi, vw in enumerate(views):
            residual = (vw["rot"] & (1 - vw["covered"])).astype(np.uint8)
            if residual.sum() == 0:
                continue
            area, top, left, h, w = largest_rectangle_in_binary_matrix(residual)
            if area <= 0:
                continue
            if (best is None) or (area > best[0]):
                best = (area, vi, (top, left, h, w))
        if best is None:
            break

        _, vi, (top, left, h, w) = best
        vw = views[vi]

        # rect in rotated coords -> polygon in small/orig coords
        x0, y0 = float(left), float(top)
        x1, y1 = x0 + w, y0 + h
        box_rot = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
        ones = np.ones((4,1), dtype=np.float32)
        box_small = np.hstack([box_rot, ones]) @ vw["Minv"].T
        box_small = order_quad_clockwise(box_small)

        # redundancy check
        redundant = False
        for *_unused, prev_box in placed_small:
            if quad_iou(prev_box, box_small) > iou_cap:
                redundant = True; break
        if redundant:
            cx_s, cy_s = float(box_small[:,0].mean()), float(box_small[:,1].mean())
            cv2.circle(vw["covered"], (int(round(cx_s)), int(round(cy_s))), 3, 1, -1)
            continue

        # accept
        w_map = float(np.linalg.norm(box_small[1] - box_small[0]))
        h_map = float(np.linalg.norm(box_small[2] - box_small[1]))
        ang = float(views[vi]["theta"])
        cx_s, cy_s = float(box_small[:,0].mean()), float(box_small[:,1].mean())
        placed_small.append((cx_s, cy_s, w_map, h_map, ang, box_small.copy()))

        cv2.fillConvexPoly(covered_small, box_small.astype(np.int32), 1)

        # update all views' covered
        for vj in views:
            box_r = np.hstack([box_small, ones]) @ vj["M"].T
            cv2.fillConvexPoly(vj["covered"], box_r.astype(np.int32), 1)

    out = []
    inv_scale = 1.0/scale
    for cx_s, cy_s, w_s, h_s, ang, box_s in placed_small:
        if scale != 1.0:
            box = (box_s * inv_scale).astype(np.float32)
            cx = cx_s * inv_scale; cy = cy_s * inv_scale
            w  = w_s  * inv_scale; h  = h_s  * inv_scale
        else:
            box = box_s.astype(np.float32); cx = cx_s; cy = cy_s; w = w_s; h = h_s
        out.append((cx, cy, w, h, ang, order_quad_clockwise(box)))
    return out

# ---------- contamination ----------
def is_contaminated_gapwise(poly_mask: np.ndarray, lev_img: np.ndarray, lvl: int) -> bool:
    total = int(poly_mask.sum())
    if total == 0:
        return False
    for lower in range(lvl-1, -1, -1):
        gap = lvl - lower
        cnt = int((poly_mask & (lev_img == lower)).sum())
        frac_th = GAP_TOL_FRAC.get(gap, 0.0)
        minpx   = GAP_TOL_MIN_PX.get(gap, 0)
        if cnt > max(minpx, int(frac_th * total)):
            return True
    return False

# ---------- Emitters ----------
ann_id = 1  # global increment
def emit_obb(box: np.ndarray, cx: float, cy: float, bw: float, bh: float, angle: float, lvl: int,
             W: int, H: int, coco_img_id: int, catid_by_level: dict, annotations_coco: list, yolo_lines: list):
    global ann_id
    # order + clip
    box = order_quad_clockwise(box.copy())
    box[:, 0] = np.clip(box[:, 0], 0, W-1)
    box[:, 1] = np.clip(box[:, 1], 0, H-1)

    # COCO AABB
    x_min, y_min = float(np.min(box[:,0])), float(np.min(box[:,1]))
    x_max, y_max = float(np.max(box[:,0])), float(np.max(box[:,1]))
    bbox_xywh = [x_min, y_min, x_max - x_min, y_max - y_min]

    annotations_coco.append({
        "id": ann_id,
        "image_id": coco_img_id,
        "category_id": catid_by_level[lvl],
        "bbox": [round(v, 2) for v in bbox_xywh],
        "area": round(float(bw * bh), 2),
        "iscrowd": 0,
        "segmentation": [box.reshape(-1).round(2).tolist()],
        "bbox_obb": [round(float(cx),2), round(float(cy),2), round(float(bw),2), round(float(bh),2), round(float(angle),2)]
    })
    ann_id += 1

    # YOLOv8-OBB (normalized corner quad)
    poly_norm = box.copy()
    poly_norm[:,0] /= W
    poly_norm[:,1] /= H
    cls_idx = catid_by_level[lvl] - 1
    yolo_lines.append(f"{cls_idx} " + " ".join(f"{poly_norm[i,0]:.6f} {poly_norm[i,1]:.6f}" for i in range(4)))

# ---------- main ----------
def main():
    ensure_dirs()

    # categories
    images_coco, annotations_coco, categories_coco = [], [], []
    level_ids = sorted([l for l in LEVEL_TO_CAT if (l >= MIN_LEVEL) or (INCLUDE_LEVEL1 and l == 1)])
    for i, lvl in enumerate(level_ids, start=1):
        categories_coco.append({"id": i, "name": LEVEL_TO_CAT[lvl], "supercategory": "slz"})
    catid_by_level = {lvl: i+1 for i, lvl in enumerate(level_ids)}

    # GSD
    if not ALTS_CSV.exists():
        raise FileNotFoundError(f"Missing {ALTS_CSV}. Reuse the same file you used to compute masks.")
    gsd_by_id = load_GSDs(ALTS_CSV)

    lev_paths = sorted(LEV_DIR.glob("*_levels.png"))
    print(f"Found {len(lev_paths)} level masks")

    for lev_path in tqdm(lev_paths, desc="SLZ OBB labels"):
        stem = lev_path.stem  # '012_levels'
        img_id = stem.replace("_levels", "")

        # image path
        img_path = IM_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = IM_DIR / f"{img_id}.png"
            if not img_path.exists():
                # skip if base image missing; still can export labels with dimensions from mask
                img_path = None

        # load levels
        try:
            lev = load_levels_mask(lev_path)
        except Exception as e:
            print(f"[warn] cannot load {lev_path}: {e}")
            continue

        # size / base image
        if img_path is not None:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            H, W = (img.shape[0], img.shape[1]) if img is not None else lev.shape[:2]
        else:
            img = None
            H, W = lev.shape[:2]

        # COCO image record
        coco_img_id = len(images_coco) + 1
        file_name = img_path.name if img_path is not None else lev_path.name
        images_coco.append({"id": coco_img_id, "file_name": file_name, "width": int(W), "height": int(H)})

        # per-image min area threshold
        s = gsd_by_id.get(img_id)
        A_min_px = amin_pixels(D_M, s, factor=1.5) if s is not None else 0

        # scratch poly buffer
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        yolo_lines: List[str] = []
        viz = None
        if WRITE_VIZ and img is not None:
            viz = img.copy()

        # process levels high→low
        levels_to_export = [3, 2] + ([1] if INCLUDE_LEVEL1 else [])
        for lvl in levels_to_export:
            if lvl < MIN_LEVEL and not (INCLUDE_LEVEL1 and lvl == 1):
                continue
            m = (lev == lvl).astype(np.uint8)
            if m.sum() == 0:
                continue

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for cid in range(1, num):
                x, y, w_roi, h_roi, area_roi = stats[cid]
                if area_roi < max(10, A_min_px):
                    continue
                comp = (labels[y:y+h_roi, x:x+w_roi] == cid).astype(np.uint8)

                # ---------- base estimate ----------
                # r = rect_from_mask_component(comp)
                # if r is None:
                    # continue
                # cx_r, cy_r, bw_r, bh_r, ang_r, box_roi = r
                # to image coords
                # box_img = box_roi.copy(); box_img[:,0] += x; box_img[:,1] += y
                # cx_b, cy_b = cx_r + x, cy_r + y

                # contamination check (gap-wise)
                # poly_mask[:] = 0
                # cv2.fillConvexPoly(poly_mask, box_img.astype(np.int32), 1)
                # contaminated = is_contaminated_gapwise(poly_mask, lev, lvl)

                # if not contaminated:
                #     # accept base
                #     emit_obb(box_img, cx_b, cy_b, bw_r, bh_r, ang_r, lvl, W, H, coco_img_id, catid_by_level, annotations_coco, yolo_lines)
                #     if viz is not None:
                #         cv2.polylines(viz, [box_img.astype(np.int32)], True, (0,255,0) if lvl==3 else (255,165,0), 2)
                #     continue

                # ---------- salvage ----------
                # 1) Build allowed mask (L≥lvl & CC) in ROI
                allowed_roi = build_allowed_mask_for_component(lev, lvl, (x,y,w_roi,h_roi), comp, connectivity=8)
                allowed_area = int(allowed_roi.sum())
                if allowed_area == 0:
                    continue

                # salvaged = False

                # 1) refine angle+scale (small translation) inside allowed
                # cnts, _ = cv2.findContours(allowed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # ang_hint = None
                # if cnts:
                #     (_, _), (w0,h0), a0 = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
                #     if w0 < h0: a0 += 90.0
                #     ang_hint = a0 % 180.0

                # res1 = refine_angle_scale_in_allowed_roi(cx_b, cy_b, bw_r, bh_r, ang_r,
                #                                          allowed_roi, (x,y,w_roi,h_roi),
                #                                          ang_hint=ang_hint, trans_px=2, scale_iters=12)
                # if res1 is not None:
                #     cx1, cy1, w1, h1, ang1, box1 = res1
                #     area1 = w1 * h1
                #     coverage1 = area1 / max(1, allowed_area)
                #     if (area1 >= max(10, A_min_px)) and (coverage1 >= SALVAGE_MIN_FRAC_OF_COMP):
                #         emit_obb(box1, cx1, cy1, w1, h1, ang1, lvl, W, H, coco_img_id, catid_by_level, annotations_coco, yolo_lines)
                #         if viz is not None:
                #             cv2.polylines(viz, [box1.astype(np.int32)], True, (0,128,255), 2)
                #         salvaged = True

                # if salvaged:
                #     continue

                # 2) single max-inscribed in allowed (ban ROI edges first)
                # allowed2 = zero_out_roi_frame(allowed_roi, ROI_EDGE_PAD)
                # res2 = _max_inscribed_rect_in_allowed(allowed2, angle_hint_deg=ang_r, margin_px=GUARD_PX, angle_probe=())
                # if res2 is not None:
                #     ax, ay, w2, h2, ang2, box2_roi = res2
                #     box2 = box2_roi.copy(); box2[:,0] += x; box2[:,1] += y
                #     cx2, cy2 = ax + x, ay + y
                #     area2 = w2 * h2
                #     coverage2 = area2 / max(1, allowed_area)
                #     if (area2 >= max(10, A_min_px)) and (coverage2 >= SALVAGE_MIN_FRAC_OF_COMP):
                #         emit_obb(box2, cx2, cy2, w2, h2, ang2, lvl, W, H, coco_img_id, catid_by_level, annotations_coco, yolo_lines)
                #         if viz is not None:
                #             cv2.polylines(viz, [box2.astype(np.int32)], True, (0,0,255), 2)
                #         salvaged = True

                # if salvaged:
                #     continue

                # 3) Greedy rotated max-rect cover
                angles = angle_candidates_from_mask(allowed_roi)
                rects = greedy_cover_with_rotated_maxrect_fast(
                    allowed_roi, guard_px=GUARD_PX, cover_frac=GREEDY_COVER_FRAC,
                    max_rects=GREEDY_MAX_RECTS, iou_cap=GREEDY_IOU_CAP,
                    angles=angles, max_side=GREEDY_MAX_SIDE
                )

                for cx3, cy3, w3, h3, ang3, box3_roi in rects:
                    if w3 * h3 < max(10, A_min_px):
                        continue
                    box3 = box3_roi.copy(); box3[:,0] += x; box3[:,1] += y
                    cx3_img, cy3_img = cx3 + x, cy3 + y
                    emit_obb(box3, cx3_img, cy3_img, w3, h3, ang3, lvl, W, H, coco_img_id, catid_by_level, annotations_coco, yolo_lines)
                    if viz is not None:
                        cv2.polylines(viz, [box3.astype(np.int32)], True, (0,0,255), 2)

        # write YOLO label file
        (YOLO_DIR / f"{img_id}.txt").write_text("\n".join(yolo_lines))

        # save visualization
        if WRITE_VIZ and viz is not None:
            cv2.imwrite(str(VIZ_DIR / f"{img_id}_obb.jpg"), viz)

    # Dump COCO JSON
    with open(COCO_JSON, "w") as f:
        json.dump({
            "images": images_coco,
            "annotations": annotations_coco,
            "categories": categories_coco
        }, f, indent=2)

    print(f"Done. Wrote {len(images_coco)} images, {len(annotations_coco)} annotations.")
    print(f"YOLO labels: {YOLO_DIR}")
    print(f"COCO JSON  : {COCO_JSON}")
    if WRITE_VIZ:
        print(f"Viz        : {VIZ_DIR}")

if __name__ == "__main__":
    main()
