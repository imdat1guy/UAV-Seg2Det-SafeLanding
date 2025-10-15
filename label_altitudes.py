# === Altitude Annotation Carousel (Tkinter) ===
# - Shows images one-by-one (overlay if available)
# - Prefills predicted altitude; lets you overwrite or accept
# - Saves progress to manual_altitudes.csv as you go
# - Exports altitudes_final.csv (manual overrides > predictions)

from pathlib import Path
import csv, os, json, time
from datetime import datetime
from typing import Optional, Dict
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# -----------------------------
# Paths (match your project)
# -----------------------------
DATA_ROOT  = Path("Data/training_set")
IM_DIR     = DATA_ROOT / "images"
OUT_DIR    = DATA_ROOT / "slz_out"
ALT_DIR    = OUT_DIR / "altitude"
OVERLAY_DIR= ALT_DIR / "overlays"

ALT_CSV_PRED   = ALT_DIR / "altitude_estimates.csv"    # from your estimator
ALT_CSV_MANUAL = ALT_DIR / "manual_altitudes.csv"      # this UI writes here
ALT_CSV_FINAL  = ALT_DIR / "altitudes_final.csv"       # merged on Export

ALT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utility: load predictions
# -----------------------------
def load_predictions(csv_path: Path) -> Dict[str, dict]:
    preds = {}
    if not csv_path.exists():
        return preds
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_id = r.get("image_id") or r.get("id") or r.get("filename")
            if not img_id:
                continue
            palt = r.get("est_alt_m", "") or ""
            try:
                pred_alt = float(palt) if palt != "" else None
            except ValueError:
                pred_alt = None
            preds[img_id] = {
                "pred_alt": pred_alt,
                "gsd": r.get("gsd_m_per_px", ""),
                "method": r.get("method", r.get("notes", "")),
                "raw": r
            }
    return preds

# -----------------------------
# Utility: load existing manual
# -----------------------------
def load_manual(csv_path: Path) -> Dict[str, dict]:
    manual = {}
    if not csv_path.exists():
        return manual
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_id = r.get("image_id")
            if not img_id:
                continue
            try:
                m = float(r.get("manual_alt_m", ""))
            except (TypeError, ValueError):
                m = None
            manual[img_id] = {
                "manual_alt_m": m,
                "timestamp": r.get("timestamp_iso", ""),
                "predicted_alt_m": r.get("predicted_alt_m", ""),
                "method": r.get("method", ""),
                "reviewer": r.get("reviewer", "")
            }
    return manual

# -----------------------------
# Utility: list image IDs
# -----------------------------
def list_image_ids() -> list[str]:
    return sorted(p.stem for p in IM_DIR.glob("*.jpg"))

# -----------------------------
# Utility: load display image
# -----------------------------
def load_display_image(img_id: str, prefer_overlay: bool, max_w=1280, max_h=900) -> Optional[Image.Image]:
    if prefer_overlay:
        ov = OVERLAY_DIR / f"{img_id}_alt_overlay.jpg"
        if ov.exists():
            try:
                im = Image.open(ov).convert("RGB")
                return _resize(im, max_w, max_h)
            except Exception:
                pass
    ip = IM_DIR / f"{img_id}.jpg"
    if not ip.exists():
        return None
    im = Image.open(ip).convert("RGB")
    return _resize(im, max_w, max_h)

def _resize(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = im.size
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return im

# -----------------------------
# Save a single manual row (append-or-update on disk)
# -----------------------------
def write_manual_row(image_id: str, manual_alt_m: float, pred_alt: Optional[float], method: str, reviewer=""):
    # Load existing rows
    rows = []
    exists = ALT_CSV_MANUAL.exists()
    if exists:
        with open(ALT_CSV_MANUAL, newline="") as f:
            rows = list(csv.DictReader(f))
    # Update or append
    updated = False
    now_iso = datetime.utcnow().isoformat()
    for r in rows:
        if r.get("image_id") == image_id:
            r["manual_alt_m"] = f"{manual_alt_m:.2f}"
            r["predicted_alt_m"] = "" if pred_alt is None else f"{pred_alt:.2f}"
            r["method"] = method
            r["reviewer"] = reviewer
            r["timestamp_iso"] = now_iso
            updated = True
            break
    if not updated:
        rows.append({
            "image_id": image_id,
            "manual_alt_m": f"{manual_alt_m:.2f}",
            "predicted_alt_m": "" if pred_alt is None else f"{pred_alt:.2f}",
            "method": method,
            "reviewer": reviewer,
            "timestamp_iso": now_iso
        })
    # Write back
    with open(ALT_CSV_MANUAL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id","manual_alt_m","predicted_alt_m","method","reviewer","timestamp_iso"])
        writer.writeheader()
        writer.writerows(rows)

# -----------------------------
# Export merged final CSV
# -----------------------------
def export_final_csv(preds: Dict[str,dict], manual: Dict[str,dict], img_ids: list[str]):
    rows = []
    for img_id in img_ids:
        pred_alt = preds.get(img_id, {}).get("pred_alt", None)
        method   = preds.get(img_id, {}).get("method", "")
        if img_id in manual and manual[img_id].get("manual_alt_m") is not None:
            final = manual[img_id]["manual_alt_m"]
            source = "manual"
        else:
            final = pred_alt
            source = "predicted" if pred_alt is not None else ""
        rows.append({
            "image_id": img_id,
            "final_alt_m": "" if final is None else f"{final:.2f}",
            "source": source,
            "predicted_alt_m": "" if pred_alt is None else f"{pred_alt:.2f}",
            "method": method
        })
    with open(ALT_CSV_FINAL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id","final_alt_m","source","predicted_alt_m","method"])
        writer.writeheader()
        writer.writerows(rows)
    return ALT_CSV_FINAL

# -----------------------------
# Tkinter App
# -----------------------------
class AltAnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Altitude Annotation Carousel")

        # Data
        self.img_ids = list_image_ids()
        if not self.img_ids:
            messagebox.showerror("Error", f"No images in {IM_DIR}")
            return
        self.preds  = load_predictions(ALT_CSV_PRED)
        self.manual = load_manual(ALT_CSV_MANUAL)

        # UI State
        self.idx = 0
        self.prefer_overlay = tk.BooleanVar(value=True)
        self.reviewer = tk.StringVar(value="")
        self.jump_id_var = tk.StringVar(value="")
        self.alt_var = tk.StringVar(value="")

        # Layout
        self._build_ui()

        # Start with first image (or first unreviewed if preferred)
        self._load_current()

        # Keybindings
        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<Right>",  lambda e: self.next_image())
        self.root.bind("<Left>",   lambda e: self.prev_image())
        self.root.bind("<Escape>", lambda e: self.root.quit())

    def _build_ui(self):
        self.root.geometry("1480x980")

        # Left: image
        left = ttk.Frame(self.root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.img_label = ttk.Label(left)
        self.img_label.pack(fill=tk.BOTH, expand=True)

        # Right: controls
        right = ttk.Frame(self.root, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.lbl_id = ttk.Label(right, text="image_id", font=("TkDefaultFont", 12, "bold"))
        self.lbl_id.pack(anchor="w", pady=(0,4))

        self.lbl_pred = ttk.Label(right, text="Predicted: —")
        self.lbl_pred.pack(anchor="w", pady=(0,8))

        # Alt entry
        frm_alt = ttk.Frame(right)
        frm_alt.pack(anchor="w", pady=(0,10))
        ttk.Label(frm_alt, text="Altitude (m): ").pack(side=tk.LEFT)
        self.entry_alt = ttk.Entry(frm_alt, textvariable=self.alt_var, width=12)
        self.entry_alt.pack(side=tk.LEFT)

        # Buttons
        btns1 = ttk.Frame(right); btns1.pack(anchor="w", pady=(8,4))
        ttk.Button(btns1, text="Save & Next (Enter)", command=self.save_and_next).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns1, text="Prev (←)", command=self.prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns1, text="Next (→)", command=self.next_image).pack(side=tk.LEFT, padx=4)

        btns2 = ttk.Frame(right); btns2.pack(anchor="w", pady=(8,4))
        ttk.Button(btns2, text="Jump to first unreviewed", command=self.jump_unreviewed).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns2, text="Revert to predicted", command=self.revert_to_pred).pack(side=tk.LEFT, padx=4)

        # Options
        opts = ttk.Frame(right); opts.pack(anchor="w", pady=(8,4))
        ttk.Checkbutton(opts, text="Prefer overlay", variable=self.prefer_overlay, command=self._refresh_image).pack(side=tk.LEFT, padx=4)
        ttk.Label(opts, text="Reviewer:").pack(side=tk.LEFT, padx=(12,4))
        ttk.Entry(opts, textvariable=self.reviewer, width=14).pack(side=tk.LEFT)

        # Jump to ID
        jmp = ttk.Frame(right); jmp.pack(anchor="w", pady=(8,4))
        ttk.Label(jmp, text="Jump to ID:").pack(side=tk.LEFT, padx=(0,4))
        ttk.Entry(jmp, textvariable=self.jump_id_var, width=10).pack(side=tk.LEFT)
        ttk.Button(jmp, text="Go", command=self.jump_to_id).pack(side=tk.LEFT, padx=4)

        # Progress + export
        self.lbl_prog = ttk.Label(right, text="0 / 0 reviewed")
        self.lbl_prog.pack(anchor="w", pady=(12,8))
        ttk.Button(right, text="Export final CSV", command=self.export_final).pack(anchor="w", pady=(4,4))

        # Hint
        ttk.Label(right, text="Shortcuts: Enter=Save&Next, ←/→=Prev/Next, Esc=Quit").pack(anchor="w", pady=(12,0))

    def _refresh_progress(self):
        reviewed = sum(1 for i in self.img_ids if self.manual.get(i, {}).get("manual_alt_m") is not None)
        self.lbl_prog.config(text=f"Reviewed {reviewed} / {len(self.img_ids)}")

    def _load_current(self):
        img_id = self.img_ids[self.idx]
        self.lbl_id.config(text=f"{img_id}.jpg")
        pred_alt = self.preds.get(img_id, {}).get("pred_alt", None)
        method   = self.preds.get(img_id, {}).get("method", "")
        self.lbl_pred.config(text=f"Predicted: {'—' if pred_alt is None else f'{pred_alt:.2f} m'}  ({method or 'n/a'})")

        # Prefill box with manual if exists, else predicted
        if img_id in self.manual and self.manual[img_id].get("manual_alt_m") is not None:
            self.alt_var.set(f"{self.manual[img_id]['manual_alt_m']:.2f}")
        else:
            self.alt_var.set("" if pred_alt is None else f"{pred_alt:.2f}")

        self._refresh_image()
        self._refresh_progress()
        self.entry_alt.focus_set()
        self.entry_alt.select_range(0, tk.END)

    def _refresh_image(self):
        img_id = self.img_ids[self.idx]
        pil = load_display_image(img_id, self.prefer_overlay.get())
        if pil is None:
            canvas = Image.new("RGB", (800, 600), (30,30,30))
            self.tkimg = ImageTk.PhotoImage(canvas)
        else:
            self.tkimg = ImageTk.PhotoImage(pil)
        self.img_label.configure(image=self.tkimg)

    def save_and_next(self):
        img_id = self.img_ids[self.idx]
        txt = self.alt_var.get().strip()
        if txt == "":
            if not messagebox.askyesno("Confirm", "Altitude box is empty. Save as blank and continue?"):
                return
            manual_alt = None
        else:
            try:
                manual_alt = float(txt)
                if manual_alt < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid value", "Enter a non-negative number (meters), or leave blank.")
                return

        pred_alt = self.preds.get(img_id, {}).get("pred_alt", None)
        method   = self.preds.get(img_id, {}).get("method", "")
        # Write if a number; if blank, we still record a blank row to mark reviewed? (optional)
        if manual_alt is not None:
            write_manual_row(img_id, manual_alt, pred_alt, method, reviewer=self.reviewer.get().strip())
            # also update in-memory manual
            self.manual[img_id] = {
                "manual_alt_m": manual_alt,
                "timestamp": datetime.utcnow().isoformat(),
                "predicted_alt_m": "" if pred_alt is None else f"{pred_alt:.2f}",
                "method": method,
                "reviewer": self.reviewer.get().strip()
            }
        else:
            # treat as "skipped" (do not write or remove any prior manual)
            pass

        # Move next
        self.next_image()

    def revert_to_pred(self):
        img_id = self.img_ids[self.idx]
        pred_alt = self.preds.get(img_id, {}).get("pred_alt", None)
        self.alt_var.set("" if pred_alt is None else f"{pred_alt:.2f}")
        self.entry_alt.focus_set()
        self.entry_alt.select_range(0, tk.END)

    def next_image(self):
        if self.idx < len(self.img_ids) - 1:
            self.idx += 1
            self._load_current()

    def prev_image(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_current()

    def jump_unreviewed(self):
        for j, iid in enumerate(self.img_ids):
            if self.manual.get(iid, {}).get("manual_alt_m") is None:
                self.idx = j
                self._load_current()
                return
        messagebox.showinfo("All done", "No unreviewed images found.")

    def jump_to_id(self):
        target = self.jump_id_var.get().strip()
        if target.endswith(".jpg"):
            target = target[:-4]
        if target in self.img_ids:
            self.idx = self.img_ids.index(target)
            self._load_current()
        else:
            messagebox.showerror("Not found", f"{target} not in images.")

    def export_final(self):
        out = export_final_csv(self.preds, self.manual, self.img_ids)
        messagebox.showinfo("Exported", f"Final CSV written:\n{out}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    # Quick sanity check
    if not IM_DIR.exists():
        raise SystemExit(f"Images dir not found: {IM_DIR}")
    if not ALT_CSV_PRED.exists():
        print(f"Warning: predictions CSV not found: {ALT_CSV_PRED}\n"
              f"You can still annotate manually; predicted boxes will be blank.")

    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = AltAnnotatorApp(root)
    root.mainloop()
