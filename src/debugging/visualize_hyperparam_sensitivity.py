#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter sensitivity study for the safety-map generator.

Sweeps lambda, beta, eps, tau0, tau1 (one at a time, others fixed at the
values used in `exploring.ipynb`) on image 243 from the ICG Semantic Drone
Dataset and saves one figure per knob. Each figure has 4 panels in a row:
the original RGB image plus three RGB+safety-mask overlays. The panel
whose value matches the published "chosen" value is labelled (chosen).

All safety-map functions are copied verbatim from `exploring.ipynb` so the
study reflects the exact pipeline used in the paper.

Run from the repo root:
    python src/debugging/visualize_hyperparam_sensitivity.py
"""

from pathlib import Path
from dataclasses import dataclass
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------ CONFIG ------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DATA_ROOT = REPO_ROOT / "Data" / "training_set"
IM_DIR    = DATA_ROOT / "images"
LAB_DIR   = DATA_ROOT / "gt" / "semantic" / "label_images"
CSV_PATH  = DATA_ROOT / "gt" / "semantic" / "class_dict.csv"
ALTS_CSV  = DATA_ROOT / "slz_out" / "altitude" / "altitudes_final.csv"

IMG_ID    = "243"

OUT_DIR   = HERE / "hyperparam_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Camera params (Sony a6000)
IMG_W            = 6000
SENSOR_W_MM      = 23.5
ASSUMED_FOCAL_MM = 16.0
PX_SIZE_MM       = SENSOR_W_MM / IMG_W

# Aircraft + safety config (must match exploring.ipynb)
D_M             = 0.40
DELTA_POS_M     = 0.8
LAMBDA_STANDOFF = 3

# Chosen defaults from exploring.ipynb
DEFAULTS = dict(lam=3, beta=0.9, eps=0.10, tau0=1.20, tau1=0.60)

# Sweep values per knob (chosen value must appear in the list)
SWEEPS = [
    # (symbol_for_title, kwarg_name, values, out_filename)
    (r"$\lambda$",    "lam",  [1,    3,    5  ], "sensitivity_lambda.png"),
    (r"$\beta$",      "beta", [0.5,  0.9,  1.0], "sensitivity_beta.png"),
    (r"$\varepsilon$","eps",  [0.01, 0.10, 1.0], "sensitivity_eps.png"),
    (r"$\tau_0$",     "tau0", [0.6,  1.2,  2.4], "sensitivity_tau0.png"),
    (r"$\tau_1$",     "tau1", [0.3,  0.6,  1.2], "sensitivity_tau1.png"),
]
# --------------------------------------------


# ===== Base safety categories =====
H_SAFE, SAFE, UNSAFE, HAZARD = 0, 1, 2, 3

BASE_BY_NAME = {
    "grass": H_SAFE, "dirt": H_SAFE, "gravel": H_SAFE, "unlabeled": H_SAFE,
    "paved-area": SAFE, "roof": SAFE, "ar-marker": SAFE,
    "rocks": UNSAFE, "wall": UNSAFE, "fence": UNSAFE, "fence-pole": UNSAFE,
    "vegetation": UNSAFE, "tree": UNSAFE, "bald-tree": UNSAFE, "obstacle": UNSAFE,
    "water": UNSAFE, "pool": UNSAFE, "window": UNSAFE, "door": UNSAFE,
    "person": HAZARD, "dog": HAZARD, "car": HAZARD, "bicycle": HAZARD,
    "conflicting": HAZARD,
}


@dataclass
class SafetyParams:
    beta: float = 0.9
    tau1: float = 0.60
    tau0: float = 1.20
    eps:  float = 0.10


# ===== Class table & lookups =====
def load_class_table(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f, skipinitialspace=True)):
            name = row["name"].strip().lower()
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            rows.append({"cid": i, "name": name, "rgb": (r, g, b)})

    rgb2cid = np.full((256**3,), 255, dtype=np.uint16)
    for r in rows:
        code = (r["rgb"][0] << 16) | (r["rgb"][1] << 8) | r["rgb"][2]
        rgb2cid[code] = r["cid"]
    cid2name = {r["cid"]: r["name"] for r in rows}
    cid2base = {cid: BASE_BY_NAME[name] for cid, name in cid2name.items() if name in BASE_BY_NAME}
    return rgb2cid, cid2name, cid2base


# ===== Helper functions (verbatim from exploring.ipynb) =====
def compute_margins(D_m: float, delta_pos_m: float, lam: float):
    d2 = D_m / 2.0 + delta_pos_m
    d1 = d2 + lam * D_m
    return d1, d2


def amin_pixels(D_m: float, s_m_per_px: float, factor: float = 1.5) -> int:
    area_m2 = factor * (np.pi * (D_m / 2.0) ** 2)
    return int(np.ceil(area_m2 / (s_m_per_px ** 2)))


def euclidean_dt(binary: np.ndarray, s_m_per_px: float) -> np.ndarray:
    inv = (1 - (binary > 0).astype(np.uint8)) * 255
    dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return dist_px * s_m_per_px


def compute_risk(DH: np.ndarray, DU: np.ndarray, beta: float, eps: float) -> np.ndarray:
    return (beta / (DH + eps)) + ((1.0 - beta) / (DU + eps))


def enforce_min_area(levels: np.ndarray, level: int, A_min_px: int) -> None:
    tgt = (levels == level).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(tgt, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < A_min_px:
            levels[lab == i] = 0


def png_mask_to_class_ids(mask_bgr: np.ndarray, rgb2cid: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    codes = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)
    return rgb2cid[codes]


def safety_map_from_class_mask(cls_mask: np.ndarray, s_m_per_px: float,
                               d1_m: float, d2_m: float, p: SafetyParams,
                               cid2base: dict):
    base = np.full_like(cls_mask, -1, dtype=np.int16)
    for cid, base_cat in cid2base.items():
        base[cls_mask == cid] = base_cat

    H = (base == HAZARD).astype(np.uint8)
    U = (base == UNSAFE).astype(np.uint8)

    DH = euclidean_dt(H, s_m_per_px)
    DU = euclidean_dt(U, s_m_per_px)

    DHp = np.maximum(DH - D_M, 0.0)
    minDp = np.minimum(DHp, DU)

    R = compute_risk(DH, DU, p.beta, p.eps)

    D1_M, D2_M = compute_margins(D_M, DELTA_POS_M, LAMBDA_STANDOFF)

    L = np.zeros_like(cls_mask, dtype=np.uint8)
    Hs = (base == H_SAFE); Sa = (base == SAFE); Un = (base == UNSAFE); Hz = (base == HAZARD)

    L[ Hs & (DHp >= d1_m) & (DU >= d1_m) ] = 3
    L[ (Hs & (minDp >= D2_M) & (minDp < D1_M)) | (Sa & (minDp >= D2_M)) ] = 2
    L[ (Sa | Hs | Un) & ((minDp < d2_m) | (R > p.tau1)) ] = 1
    L[ Hz | (R >= p.tau0) ] = 0

    return L.astype(np.uint8), R.astype(np.float32), DHp.astype(np.float32), DU.astype(np.float32), base


def colorize_levels(levels: np.ndarray) -> np.ndarray:
    cmap = {0: (30, 30, 220), 1: (0, 165, 255), 2: (255, 165, 0), 3: (0, 180, 0)}
    out = np.zeros((*levels.shape, 3), dtype=np.uint8)
    for k, c in cmap.items():
        out[levels == k] = c
    return out


def edge_smooth_promote_safe_adjacent(L: np.ndarray, base: np.ndarray,
                                      DHp: np.ndarray, DUp: np.ndarray,
                                      grow_frac: float = 0.05) -> np.ndarray:
    L3 = (L == 3).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(L3, connectivity=8)
    if num <= 1:
        return L
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(1, num):
        comp = (lab == i)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        ring = cv2.dilate(comp.astype(np.uint8), se) & (~comp.astype(np.uint8))
        cand = (ring.astype(bool)
                & (base == SAFE)
                & (L != 0)
                & (np.minimum(DHp, DUp) > 0))
        cand_area = int(np.count_nonzero(cand))
        if cand_area == 0:
            continue
        if cand_area <= grow_frac * area:
            L[cand] = 3
    return L


def gsd_from_altitude_m(alt_m: float) -> float:
    return alt_m * (PX_SIZE_MM / ASSUMED_FOCAL_MM)


# ===== Data loading =====
def load_altitude_for(img_id: str, csv_path: Path) -> float:
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if row["image_id"].strip() == img_id:
                return float(row["final_alt_m"])
    raise KeyError(f"image_id {img_id!r} not found in {csv_path}")


def load_image_and_mask(img_id: str, rgb2cid: np.ndarray):
    img_path = IM_DIR / f"{img_id}.jpg"
    lab_path = LAB_DIR / f"{img_id}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    lab = cv2.imread(str(lab_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    if lab is None:
        raise FileNotFoundError(lab_path)
    cls_mask = png_mask_to_class_ids(lab, rgb2cid)
    alt = load_altitude_for(img_id, ALTS_CSV)
    gsd = gsd_from_altitude_m(alt)
    return img, cls_mask, alt, gsd


# ===== Pipeline wrapper =====
def run_with_overrides(cls_mask, gsd, cid2base, *,
                       lam=DEFAULTS["lam"], beta=DEFAULTS["beta"],
                       eps=DEFAULTS["eps"], tau0=DEFAULTS["tau0"],
                       tau1=DEFAULTS["tau1"]):
    d1, d2 = compute_margins(D_M, DELTA_POS_M, lam)
    p = SafetyParams(beta=beta, tau1=tau1, tau0=tau0, eps=eps)
    L, R, DHp, DU, base = safety_map_from_class_mask(cls_mask, gsd, d1, d2, p, cid2base)
    L = edge_smooth_promote_safe_adjacent(L, base, DHp, DU, grow_frac=0.05)
    a_min_px = amin_pixels(D_M, gsd, factor=1.5)
    enforce_min_area(L, 3, a_min_px)
    enforce_min_area(L, 2, a_min_px)
    return L


# ===== Plotting =====
def overlay_levels_on_image(img_bgr: np.ndarray, L: np.ndarray) -> np.ndarray:
    lev_color = colorize_levels(L)
    overlay = cv2.addWeighted(img_bgr, 1.0, lev_color, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def plot_sweep(symbol: str, kwarg: str, values, chosen, img_bgr, cls_mask, gsd,
               cid2base, out_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=22)
    axes[0].axis("off")

    for ax, val in zip(axes[1:], values):
        L = run_with_overrides(cls_mask, gsd, cid2base, **{kwarg: val})
        ax.imshow(overlay_levels_on_image(img_bgr, L))
        suffix = "  (chosen)" if val == chosen else ""
        ax.set_title(f"{symbol} = {val}{suffix}", fontsize=22)
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=(220/255, 30/255, 30/255),  label="L0 — Unsafe"),
        mpatches.Patch(color=(255/255, 165/255, 0/255),  label="L1 — Caution"),
        mpatches.Patch(color=(0/255, 165/255, 255/255),  label="L2 — Safe"),
        mpatches.Patch(color=(0/255, 180/255, 0/255),    label="L3 — Very Safe"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=18, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Output dir: {OUT_DIR}")
    rgb2cid, cid2name, cid2base = load_class_table(CSV_PATH)
    print(f"Loaded {len(cid2name)} classes ({len(cid2base)} mapped to safety categories)")

    img_bgr, cls_mask, alt, gsd = load_image_and_mask(IMG_ID, rgb2cid)
    print(f"Image {IMG_ID}: shape={img_bgr.shape}, altitude={alt:.2f} m, GSD={gsd:.6f} m/px")

    for symbol, kwarg, values, fname in SWEEPS:
        out_path = OUT_DIR / fname
        chosen = DEFAULTS[kwarg]
        print(f"  sweeping {kwarg} over {values} (chosen={chosen}) -> {out_path.name}")
        plot_sweep(symbol, kwarg, values, chosen, img_bgr, cls_mask, gsd,
                   cid2base, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
