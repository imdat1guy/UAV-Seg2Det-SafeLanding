#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the three salvage stages (S1, S2, S3) for the first contaminated OBB
on a given image. Figures are drawn over the allowed ROI with edge-ban + guard.

Matches the behavior of:
 - S1: refine_angle_scale_in_allowed_roi (returns IMAGE-coord boxes)
 - S2: _max_inscribed_rect_in_allowed with free angle set (like your separate
       max-inscribed visualizer) to recover the 90° winner
 - S3: greedy_cover_with_rotated_maxrect_fast

Edit the CONFIG paths below if needed.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
LEVELS_PNG = Path("../../Data/training_set/slz_out/masks_levels/146_levels.png")
CONT_JSON  = Path("obbs_contaminated_146.json")
CONT_INDEX = 0         # which contaminated OBB to visualize from the JSON
LEVEL      = 3

OUT_DIR = Path("salvage_vis")  # <--- add this
OUT_DIR.mkdir(exist_ok=True)

# Salvage knobs (must match your pipeline)
ROI_EDGE_PAD = 1
GUARD_PX     = 2
SALVAGE_MIN_FRAC_OF_COMP = 0.70

# S2 search angles (compact, same as your max-inscribed visualizer)
ANGLE_PROBE_S2 = (+15, -15, +30, -30)
ITERS_S2       = 25

# Greedy knobs (must match your pipeline)
GREEDY_COVER_FRAC = 0.80
GREEDY_MAX_RECTS  = 12
GREEDY_IOU_CAP    = 0.50
GREEDY_MAX_SIDE   = 1000
# ---------------------------------------------------------------

# Put these near the top of your script
FIG_SIZE = (6.2, 4.4)   # inches; pick any size you like
TOP = 0.95              # leave room for the title; 1.0 means no room
BORDERLESS = dict(left=0, right=1, bottom=0, top=TOP)

def save_square(fig, path):
    fig.set_size_inches(*FIG_SIZE, forward=True)
    fig.subplots_adjust(**BORDERLESS)
    fig.savefig(path, dpi=600)           # <-- no bbox_inches="tight"
    plt.close(fig)


# ---------- shared helpers (borrowed from your exporter) ----------
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

def rect_from_contour(cnt: np.ndarray):
    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    if w < h:
        w, h = h, w
        angle += 90.0
    angle = angle % 180.0
    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    return cx, cy, w, h, angle, order_quad_clockwise(box)

def load_levels_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    if img.ndim == 2:
        lev = img
    else:
        # OpenCV BGR/BGRA => use R channel
        lev = img[:,:,2]
    return lev.astype(np.uint8)

def build_allowed_mask_for_component(lev_img: np.ndarray,
                                     lvl: int,
                                     roi_xywh: tuple[int,int,int,int],
                                     seed_comp: np.ndarray,
                                     connectivity: int = 8) -> np.ndarray:
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
    if not nz.any():
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

def _poly_inside_mask_roi(box_roi: np.ndarray, mask_roi: np.ndarray) -> bool:
    scratch = np.zeros_like(mask_roi, dtype=np.uint8)
    cv2.fillConvexPoly(scratch, box_roi.astype(np.int32), 1)
    return int((scratch & (1 - mask_roi)).sum()) == 0

def refine_angle_scale_in_allowed_roi(cx, cy, w, h, angle0,
                                      allowed_roi: np.ndarray,
                                      roi_xywh: tuple[int,int,int,int],
                                      ang_hint,
                                      ang_search: tuple[float,...] = (-90,-45,-20,-10,0,+10,+20,+45,+90),
                                      trans_px: int = 2,
                                      scale_iters: int = 12,
                                      early_stop_pct: float = 0.02):
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
                                   angle_hint_deg,
                                   margin_px: int = 1,
                                   angle_probe: tuple[float, ...] = (+15, -15, +30, -30),
                                   iters: int = 25):
    m = (allowed_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    m_in = m if margin_px <= 0 else cv2.erode(m, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*margin_px+1, 2*margin_px+1)))
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

# ---------- fast rotated greedy cover ----------
def largest_rectangle_in_binary_matrix(bin01: np.ndarray):
    H, W = bin01.shape
    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for r in range(H):
        row = bin01[r]
        heights = np.where(row > 0, heights + 1, 0)
        stack = []; i = 0
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
    H, W = mask01.shape[:2]
    center = (W * 0.5, H * 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nW = int(H * sin + W * cos)
    nH = int(H * cos + W * sin)
    M[0,2] += (nW * 0.5) - center[0]
    M[1,2] += (nH * 0.5) - center[1]
    rot = cv2.warpAffine(mask01.astype(np.uint8), M, (nW, nH),
                         flags=cv2.INTER_NEAREST, borderValue=0)
    Minv = cv2.invertAffineTransform(M)
    return (rot > 0).astype(np.uint8), M, Minv

def angle_candidates_from_mask2(mask01: np.ndarray):
    cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [0.0, 90.0]
    (_, _), (w0,h0), ang0 = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if w0 < h0: ang0 += 90.0
    ang0 = ang0 % 180.0
    S = {0.0, 90.0, ang0, (ang0+10)%180.0, (ang0-10)%180.0}
    return sorted(S, key=lambda a: abs(((a+90)%180)-90))

def quad_iou(A: np.ndarray, B: np.ndarray) -> float:
    inter_area, _ = cv2.intersectConvexConvex(A.astype(np.float32), B.astype(np.float32))
    if inter_area <= 0: return 0.0
    aA = abs(cv2.contourArea(A)); aB = abs(cv2.contourArea(B))
    return float(inter_area / (aA + aB - inter_area + 1e-6))

def greedy_cover_with_rotated_maxrect_fast(allowed_mask: np.ndarray,
                                           guard_px: int = 2,
                                           cover_frac: float = 0.80,
                                           max_rects: int = 12,
                                           iou_cap: float = 0.50,
                                           angles=None,
                                           max_side: int = 1000):
    m = (allowed_mask > 0).astype(np.uint8)
    if m.sum() == 0: return []
    if guard_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*guard_px+1, 2*guard_px+1))
        m = cv2.erode(m, k)
        if m.sum() == 0: return []
    H, W = m.shape
    scale = 1.0; ms = m
    if max(H, W) > max_side:
        scale = max_side / float(max(H, W))
        ms = cv2.resize(m, (int(round(W*scale)), int(round(H*scale))), interpolation=cv2.INTER_NEAREST)
    if angles is None: angles = angle_candidates_from_mask2(ms)
    views = []
    for theta in angles:
        rot, M, Minv = rotate_mask_keep_all(ms, -theta)
        views.append({"theta": float(theta), "rot": rot,
                      "covered": np.zeros_like(rot, np.uint8), "M": M, "Minv": Minv})
    placed_small = []; covered_small = np.zeros_like(ms, np.uint8); total = int(ms.sum())
    for _ in range(max_rects):
        if covered_small.sum() / max(1, total) >= cover_frac: break
        best = None
        for vi, vw in enumerate(views):
            residual = (vw["rot"] & (1 - vw["covered"])).astype(np.uint8)
            if residual.sum() == 0: continue
            area, top, left, h, w = largest_rectangle_in_binary_matrix(residual)
            if area <= 0: continue
            if (best is None) or (area > best[0]): best = (area, vi, (top, left, h, w))
        if best is None: break
        _, vi, (top, left, h, w) = best; vw = views[vi]
        x0, y0 = float(left), float(top); x1, y1 = x0 + w, y0 + h
        box_rot = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
        ones = np.ones((4,1), dtype=np.float32)
        box_small = np.hstack([box_rot, ones]) @ vw["Minv"].T
        box_small = order_quad_clockwise(box_small)

        # IoU pruning
        redundant = any(quad_iou(prev_box, box_small) > iou_cap for *_unused, prev_box in placed_small)
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
        for vj in views:
            box_r = np.hstack([box_small, ones]) @ vj["M"].T
            cv2.fillConvexPoly(vj["covered"], box_r.astype(np.int32), 1)

    out = []; inv_scale = 1.0/scale
    for cx_s, cy_s, w_s, h_s, ang, box_s in placed_small:
        if scale != 1.0:
            box = (box_s * inv_scale).astype(np.float32)
            cx = cx_s * inv_scale; cy = cy_s * inv_scale
            w  = w_s  * inv_scale; h  = h_s  * inv_scale
        else:
            box = box_s.astype(np.float32); cx = cx_s; cy = cy_s; w = w_s; h = h_s
        out.append((cx, cy, w, h, ang, order_quad_clockwise(box)))
    return out
# ------------------------------------------------------------------

def main():
    # -------- load inputs --------
    lev = load_levels_mask(LEVELS_PNG)
    with open(CONT_JSON, "r") as f:
        cont = json.load(f)
    if not cont.get("obbs"):
        raise RuntimeError("No 'obbs' found in contaminated JSON.")
    obb = cont["obbs"][CONT_INDEX]
    lvl = int(obb["level"])  # safety level of this component

    poly = np.array(obb["poly"], dtype=np.float32).reshape(4,2)
    x, y, w_roi, h_roi = map(int, obb["roi_xywh"])

    # the seed component at this level within the ROI
    lev_roi = lev[y:y+h_roi, x:x+w_roi]
    m_lvl   = (lev_roi == lvl).astype(np.uint8)

    # choose the connected part that overlaps the contaminated OBB
    poly_roi = poly.copy(); poly_roi[:,0] -= x; poly_roi[:,1] -= y
    obb_mask_roi = np.zeros_like(m_lvl, np.uint8)
    cv2.fillConvexPoly(obb_mask_roi, poly_roi.astype(np.int32), 1)
    num, labels = cv2.connectedComponents(m_lvl, 8)
    if num > 1:
        best_lab, best_ov = 0, 0
        for lab in range(1, num):
            lab_m = (labels == lab).astype(np.uint8)
            ov = int((lab_m & obb_mask_roi).sum())
            if ov > best_ov:
                best_ov, best_lab = ov, lab
        seed = (labels == best_lab).astype(np.uint8)
    else:
        seed = m_lvl.copy()

    # allowed region: L >= lvl AND connected through L>=lvl to the seed
    allowed_roi = build_allowed_mask_for_component(lev, lvl, (x,y,w_roi,h_roi), seed, 8)
    allowed_area = int(allowed_roi.sum())

    # residual mask used by S1/S2/S3 (edge-ban + guard)
    allowed_padded = zero_out_roi_frame(allowed_roi, ROI_EDGE_PAD)
    residual = allowed_padded.copy()
    if GUARD_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*GUARD_PX+1, 2*GUARD_PX+1))
        residual = cv2.erode(residual, k)

    # ---------- Figure A: allowed with guard ----------
    figA = plt.figure(figsize=(7,6))
    plt.imshow((residual*255).astype(np.uint8), cmap="gray")
    # plt.title("Allowed ROI within Contaminated OBB")
    plt.axis("off")
    #figA.savefig(OUT_DIR / "A_allowed_roi.pdf", dpi=600, bbox_inches="tight")
    save_square(figA, OUT_DIR / "A_allowed_roi.pdf")
    plt.show()

    # ---------- Stage 1 (local refine) ----------
    # baseline (cx,cy,w,h,angle) — recover from the polygon in image coords
    cnt = poly.astype(np.float32).reshape(-1,1,2)
    cx0, cy0, w0, h0, ang0, _ = rect_from_contour(cnt)

    # orientation hint from allowed region (ROI coords)
    cnts, _ = cv2.findContours(allowed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ang_hint = None
    if cnts:
        (_, _), (tw,th), a0 = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        if tw < th: a0 += 90.0
        ang_hint = a0 % 180.0

    r1 = refine_angle_scale_in_allowed_roi(cx0, cy0, w0, h0, ang0,
                                           allowed_roi, (x,y,w_roi,h_roi),
                                           ang_hint=ang_hint, trans_px=2, scale_iters=12)

    if r1 is not None:
        cx1, cy1, w1, h1, ang1, box1_img = r1  # box is in IMAGE coords
        # shift to ROI for drawing
        box1 = box1_img.copy()
        box1[:,0] -= x; box1[:,1] -= y

        area = w1 * h1
        coverage = area / max(1, allowed_area) * 100

        img = cv2.cvtColor((residual*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.polylines(img, [box1.astype(np.int32)], True, (0, 180, 0), 30)
        figB = plt.figure(figsize=(7,6))
        plt.imshow(img[..., ::-1]); plt.axis("off")
        # plt.title(f"S1 winner  angle={ang1:.1f}°, w={w1:.1f}, h={h1:.1f}, Coverage={coverage:.1f}%")
        #figB.savefig(OUT_DIR / "B_stage1_S1.pdf", dpi=600, bbox_inches="tight")
        save_square(figB, OUT_DIR / "B_stage1_S1.pdf")
        plt.show()
    else:
        print("[S1] No valid refinement.")

    # ---------- Stage 2 (max-inscribed; free angle set like your visualizer) ----------
    r2 = _max_inscribed_rect_in_allowed(residual, angle_hint_deg=None,
                                        margin_px=0, angle_probe=ANGLE_PROBE_S2, iters=ITERS_S2)
    if r2 is not None:
        ax, ay, w2, h2, ang2, box2 = r2  # already in ROI coords

        area = w2 * h2
        coverage = area / max(1, allowed_area) * 100

        img = cv2.cvtColor((residual*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.polylines(img, [box2.astype(np.int32)], True, (0, 180, 0), 30)
        figC = plt.figure(figsize=(7,6))
        plt.imshow(img[..., ::-1]); plt.axis("off")
        # plt.title(f"S2 winner  angle={ang2:.1f}°, w={w2:.1f}, h={h2:.1f}, Coverage={coverage:.1f}%")
        #figC.savefig(OUT_DIR / "C_stage2_S2.pdf", dpi=600, bbox_inches="tight")
        save_square(figC, OUT_DIR / "C_stage2_S2.pdf")
        plt.show()
    else:
        print("[S2] No valid max-inscribed rectangle.")

    # ---------- Stage 3 (greedy rotated cover) ----------
    angles = angle_candidates_from_mask2(allowed_roi)
    rects = greedy_cover_with_rotated_maxrect_fast(
        allowed_roi, guard_px=GUARD_PX, cover_frac=GREEDY_COVER_FRAC,
        max_rects=GREEDY_MAX_RECTS, iou_cap=GREEDY_IOU_CAP,
        angles=angles, max_side=GREEDY_MAX_SIDE
    )
    img = cv2.cvtColor((residual*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    area_sum = 0.0
    for (cx, cy, bw, bh, ang, box) in rects:
        area_sum += bw*bh
        cv2.polylines(img, [box.astype(np.int32)], True, (0, 180, 0), 30)

    coverage = area_sum / max(1, allowed_area) * 100

    figD = plt.figure(figsize=(7,6))
    plt.imshow(img[..., ::-1]); plt.axis("off")
    # plt.title(f"S3 greedy  boxes={len(rects)}, Coverage={coverage:.1f}%")
    #figD.savefig(OUT_DIR / "D_stage3_S3.pdf", dpi=600, bbox_inches="tight")
    save_square(figD, OUT_DIR / "D_stage3_S3.pdf")
    plt.show()

if __name__ == "__main__":
    main()
