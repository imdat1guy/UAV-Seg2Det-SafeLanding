
"""Robust visualizer for the first greedy placement on image 146 (L3 contaminated OBB).
Handles both single-channel and BGRA/RGB level masks.

Edit CONFIG paths at the top if needed.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------ CONFIG (edit as needed) ------------------
LEVELS_PNG   = Path("../../Data/training_set/slz_out/masks_levels/146_levels.png")
CONT_JSON    = Path("obbs_contaminated_146.json")
CONT_INDEX   = 0       # first entry in contaminated JSON
LEVEL        = 3

ROI_EDGE_PAD = 1
GUARD_PX     = 2
ANGLE_PROBE  = (+15, -15, +30, -30)
ITERS        = 25
# -------------------------------------------------------------

def order_quad_clockwise(pts4: np.ndarray) -> np.ndarray:
    s = pts4.sum(axis=1)
    diff = (pts4[:,0] - pts4[:,1])
    tl = np.argmin(s); br = np.argmax(s)
    tr = np.argmin(diff); bl = np.argmax(diff)
    return pts4[[tl, tr, br, bl]]

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

def _fits_box_on_mask(cx, cy, w, h, ang_deg, mask01: np.ndarray):
    box = cv2.boxPoints(((cx, cy), (w, h), ang_deg)).astype(np.float32)
    poly = np.zeros_like(mask01, dtype=np.uint8)
    cv2.fillConvexPoly(poly, box.astype(np.int32), 1)
    ok = int((poly & (1 - mask01)).sum()) == 0
    return ok, box

def max_inscribed_rect_for_component_debug(
    comp_mask: np.ndarray,
    angle_hint_deg,
    margin_px: int = 0,
    angle_probe = ANGLE_PROBE,
    iters: int = ITERS
):
    trace = {}
    m = (comp_mask > 0).astype(np.uint8)
    trace["m"] = m.copy()
    if m.sum() == 0:
        return None, trace

    if margin_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin_px+1, 2*margin_px+1))
        m_in = cv2.erode(m, k)
        if m_in.sum() == 0:
            trace["m_in"] = m_in
            return None, trace
    else:
        m_in = m
    trace["m_in"] = m_in.copy()

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, trace
    cnt = max(cnts, key=cv2.contourArea)
    (_, _), (w0, h0), ang0 = cv2.minAreaRect(cnt)
    if w0 < h0:
        w0, h0 = h0, w0
        ang0 += 90.0
    ang0 = ang0 % 180.0

    dt = cv2.distanceTransform(m_in, cv2.DIST_L2, 5)
    ay, ax = np.unravel_index(np.argmax(dt), dt.shape)
    ax, ay = float(ax), float(ay)
    trace["dt"] = dt
    trace["center"] = (ax, ay)

    if angle_hint_deg is None:
        angle_candidates = [ang0] + [(ang0 + d) % 180.0 for d in angle_probe] + [0.0, 90.0]
    else:
        angle_candidates = [angle_hint_deg]
    trace["angle_candidates"] = angle_candidates

    def _fits(w, h, ang_deg):
        ok, box = _fits_box_on_mask(ax, ay, w, h, ang_deg, m_in)
        return ok, box

    best = None
    per_angle = []
    for ang in angle_candidates:
        lo, hi = 0.0, 1.0
        ok_box = None
        steps = []
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            ok, box = _fits(w0*mid, h0*mid, ang)
            steps.append({"scale": mid, "ok": bool(ok), "box": box})
            if ok:
                ok_box = box; lo = mid
            else:
                hi = mid
        area = (w0*lo) * (h0*lo) if ok_box is not None else 0.0
        per_angle.append({"angle": ang, "steps": steps, "final_scale": lo,
                          "final_area": area, "final_box": ok_box})
        if ok_box is not None and (best is None or area > best[0]):
            best = (area, ang, w0*lo, h0*lo, ok_box)

    trace["per_angle"] = per_angle
    if best is None:
        return None, trace

    area, ang_best, wb, hb, box = best
    result = (ax, ay, wb, hb, ang_best, order_quad_clockwise(box.copy()))
    trace["winner"] = {"angle": ang_best, "w": wb, "h": hb, "area": area}
    return result, trace

def load_levels_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    # Accept: single channel, RGB, or BGRA
    if img.ndim == 2:
        lev = img
    elif img.ndim == 3:
        # OpenCV uses BGR/BGRA ordering
        if img.shape[2] == 4:
            lev = img[:, :, 2]  # R from BGRA
        elif img.shape[2] == 3:
            lev = img[:, :, 2]  # R from BGR
        else:
            lev = img[:, :, 0]
    else:
        raise ValueError("Unsupported image shape for levels mask")
    return lev.astype(np.uint8)

def main():
    lev = load_levels_mask(LEVELS_PNG)

    with open(CONT_JSON, "r") as f:
        cont = json.load(f)
    if not cont.get("obbs"):
        raise RuntimeError("No 'obbs' found in contaminated JSON.")
    obb = cont["obbs"][CONT_INDEX]

    lvl = int(obb["level"])
    if lvl != LEVEL:
        print(f"[warn] obb level is {lvl}, LEVEL constant is {LEVEL}. Proceeding with obb level.")

    poly = np.array(obb["poly"], dtype=np.float32).reshape(4,2)
    x, y, w_roi, h_roi = map(int, obb["roi_xywh"])

    lev_roi = lev[y:y+h_roi, x:x+w_roi]
    m_lvl   = (lev_roi == lvl).astype(np.uint8)

    poly_roi = poly.copy()
    poly_roi[:,0] -= x; poly_roi[:,1] -= y
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

    allowed_roi = build_allowed_mask_for_component(lev, lvl, (x,y,w_roi,h_roi), seed, 8)

    allowed_roi_padded = zero_out_roi_frame(allowed_roi, ROI_EDGE_PAD)
    residual0 = allowed_roi_padded.copy()
    if GUARD_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*GUARD_PX+1, 2*GUARD_PX+1))
        residual0 = cv2.erode(residual0, k)

    res, trace = max_inscribed_rect_for_component_debug(
        residual0, angle_hint_deg=None, margin_px=0, angle_probe=ANGLE_PROBE, iters=ITERS
    )

    # ---------- Visuals ----------
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(m_lvl*255, cmap="gray"); plt.title("L==lvl (ROI)"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(allowed_roi*255, cmap="gray"); plt.title("allowed_roi (L≥lvl CC)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(residual0*255, cmap="gray"); plt.title("residual0 (edge-ban + guard)"); plt.axis("off")
    plt.tight_layout(); plt.show()

    dt = trace["dt"]; cx, cy = trace["center"]
    plt.figure(figsize=(6,5))
    plt.imshow(dt, cmap="magma"); plt.scatter([cx],[cy], s=64, c="w", edgecolors="k")
    plt.title(f"Distance Transform (DT) with chosen center ({cx},{cy})"); plt.axis("off"); plt.show()

    per = trace["per_angle"]
    cols = 3; rows = int(np.ceil(len(per)/cols))
    plt.figure(figsize=(cols*5, rows*4))
    for i, info in enumerate(per):
        base = cv2.cvtColor((residual0*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        steps = info["steps"]
        keep = list(range(len(steps))) if len(steps) <= 12 else list(np.linspace(0, len(steps)-1, 12).astype(int))
        for j in keep:
            st = steps[j]
            col = (0,200,0) if st["ok"] else (200,0,0)
            p = st["box"].astype(np.int32)
            cv2.polylines(base, [p], True, col, 1)
        if info["final_box"] is not None:
            cv2.polylines(base, [info["final_box"].astype(np.int32)], True, (0,128,255), 3)
        plt.subplot(rows, cols, i+1)
        plt.imshow(base[..., ::-1]); plt.axis("off")
        plt.title(f"angle {info['angle']:.1f}°\nscale={info['final_scale']:.3f}, area={int(info['final_area'])}")
    plt.tight_layout(); plt.show()

    if res is not None:
        cx0, cy0, w, h, ang, box = res
        img = cv2.cvtColor((residual0*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        best_info = max(per, key=lambda d: d["final_area"]) if per else None
        if best_info is not None:
            for st in best_info["steps"]:
                cv2.polylines(img, [st["box"].astype(np.int32)], True, (180,180,180), 1)
            cv2.polylines(img, [best_info["final_box"].astype(np.int32)], True, (255,255,0), 4)
        cv2.circle(img, (int(cx0), int(cy0)), 3, (0,0,0), -1)
        cv2.circle(img, (int(cx0), int(cy0)), 2, (255,255,255), -1)
        plt.figure(figsize=(6,6)); plt.imshow(img[..., ::-1]); plt.axis("off")
        plt.title(f"WINNER angle={ang:.1f}°, w={w:.1f}, h={h:.1f}, area={int(w*h)}")
        plt.show()

    print("[OK] Visualization complete.")

if __name__ == "__main__":
    main()
