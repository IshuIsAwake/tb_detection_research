"""
copypaste.py — synthetic TB positives by pasting real lesions into healthy lungs.

Core for the cp_exp arm (the driver is cp_exp/exp2_compositor.py).
The point is to beat the 799-positive data ceiling: TB lesions are scarce, healthy
lungs are abundant, so we recombine them.

Rules baked in here:
  * LEAKAGE — the lesion bank is cut from the 559 TRAIN positives ONLY (never
    val/test), so no test-set lesion can appear, pasted, in training.
  * PLACEMENT — anatomically-safe lung only: the ENTIRE patient-right lung plus the
    UPPER patient-left lung. The lower patient-left is the cardiac silhouette, where
    the pretrained segmenter over-reaches toward the heart — pasting there is the
    one thing we must avoid.
  * BLEND — two modes:
      "mixed"    = cv2.MIXED_CLONE over an elliptical matte. Keeps the target's
                   ribs, but on rib-heavy patches the donor's stronger rib edges
                   still bleed through (wrong rib count/direction).
      "residual" = subtract the donor's local background (inpaint the lesion area
                   from its ring) so the donor RIBS CANCEL, then ADD the leftover
                   lesion density onto the target's own anatomy. Physically sound:
                   on a radiograph densities are ~additive along the projection, so
                   a lesion is an added opacity, not a replacement of the bone
                   behind it. The delta is clipped to be ADDITIVE-ONLY (a lesion
                   only increases opacity) — raw signed delta darkens the target
                   where the lesion is darker than the smooth inpaint fill, which
                   looked like a dark blotch on big lesions. This is the fix for
                   donor-rib ghosting.

LATERALITY LANDMINE: chest X-ray sides are the PATIENT's. Patient-right lung is on
the IMAGE-LEFT (small x). So "entire right lung" = the whole image-LEFT lung;
"upper left lung" = the upper part of the image-RIGHT lung.

v1 pastes the lesion UNCHANGED (only relocated) — lesion-level augmentation is a
later ablation rung, kept out here so its effect is attributable on its own.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from yolo_common import settings as S, splits

IMG = 512


# ── lesion bank (from train positives only) ───────────────────────────────────

@dataclass
class Lesion:
    donor: str                       # source stem, e.g. tb0421
    cls: int                         # 0 Active, 1 Obsolete
    box: tuple[int, int, int, int]   # x1,y1,x2,y2 px in the donor 512 image
    crop: np.ndarray                 # (h,w,3) uint8 RGB


def _denorm(cx: float, cy: float, w: float, h: float, size: int = IMG):
    return ((cx - w / 2) * size, (cy - h / 2) * size,
            (cx + w / 2) * size, (cy + h / 2) * size)


def build_lesion_bank(split: dict | None = None, min_px: int = 8) -> list[Lesion]:
    """Crop every lesion box from the TRAIN positives. Same denorm as metrics.py."""
    split = split or splits.build_or_load()
    bank: list[Lesion] = []
    for stem in split["train_ids"]:
        lab = S.LABELS_ALL_DIR / f"{stem}.txt"
        img_path = S.TB_IMAGES / f"{stem}.png"
        if not lab.exists() or not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        for line in lab.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            c, cx, cy, w, h = line.split()
            x1, y1, x2, y2 = _denorm(float(cx), float(cy), float(w), float(h))
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(IMG, int(round(x2))), min(IMG, int(round(y2)))
            if x2 - x1 < min_px or y2 - y1 < min_px:
                continue
            bank.append(Lesion(stem, int(c), (x1, y1, x2, y2), img[y1:y2, x1:x2].copy()))
    return bank


# ── lesion-level appearance augmentation (Backlog D) ──────────────────────────
# v1/v2 relocate the SAME 851 crops unchanged, which turned out REDUNDANT with the
# champion's mosaic_mixup on the recall axis. This perturbs each crop's APPEARANCE
# (the one copy-paste axis mosaic_mixup cannot reach) so the detector can't memorise
# the fixed 851 textures. Transforms are lesion-local and anatomy-preserving; the
# matte and the YOLO box are recomputed downstream from the returned crop.

def _elastic(img: np.ndarray, rng: random.Random, alpha: float) -> np.ndarray:
    """Smooth random displacement field — deforms shape without shredding texture."""
    h, w = img.shape[:2]
    gen = np.random.default_rng(rng.getrandbits(32))
    sigma = max(4.0, 0.15 * min(h, w))
    dx = cv2.GaussianBlur(gen.standard_normal((h, w), dtype=np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(gen.standard_normal((h, w), dtype=np.float32), (0, 0), sigma)
    amp = alpha * min(h, w)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, (xx + amp * dx).astype(np.float32),
                     (yy + amp * dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def augment_lesion(lesion: Lesion, rng: random.Random, *, flip: bool = True,
                   rot_deg: float = 20.0, scale_jitter: float = 0.15,
                   contrast_jitter: float = 0.20, gamma_jitter: float = 0.15,
                   elastic: float = 0.0) -> Lesion:
    """Return a copy of `lesion` with an appearance-augmented crop (Backlog D).

    Each knob is a half-range about the identity: rot_deg=20 → ±20°, scale_jitter=
    0.15 → 0.85–1.15×, contrast/gamma likewise about 1.0. Contrast is taken about the
    crop mean (Poisson reconciles absolute brightness, so contrast is what varies a
    lesion's conspicuity — faint ↔ obvious). `elastic` is a fraction of the crop's
    short side (0 = off). Set any knob to 0 (or flip=False) to drop that rung. The
    returned Lesion carries the transformed crop and a box sized to it — the donor
    coords are not reused, only the width/height matter downstream."""
    crop = lesion.crop
    if flip and rng.random() < 0.5:
        crop = np.ascontiguousarray(crop[:, ::-1])
    h, w = crop.shape[:2]
    if rot_deg > 0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-rot_deg, rot_deg), 1.0)
        crop = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT101)
    if elastic > 0:
        crop = _elastic(crop, rng, elastic)
    if scale_jitter > 0:
        s = rng.uniform(1 - scale_jitter, 1 + scale_jitter)
        nw, nh = max(8, round(w * s)), max(8, round(h * s))
        crop = cv2.resize(crop, (nw, nh),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    f = crop.astype(np.float32)
    if contrast_jitter > 0:
        m = float(f.mean())
        f = (f - m) * rng.uniform(1 - contrast_jitter, 1 + contrast_jitter) + m
    if gamma_jitter > 0:
        f = 255.0 * np.clip(f / 255.0, 0.0, 1.0) ** rng.uniform(1 - gamma_jitter, 1 + gamma_jitter)
    crop = np.clip(f, 0, 255).astype(np.uint8)
    nh, nw = crop.shape[:2]
    return Lesion(lesion.donor, lesion.cls, (0, 0, nw, nh), np.ascontiguousarray(crop))


# ── placement region: entire right lung + upper left lung ─────────────────────

def split_lungs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(right, left) bool masks. right = patient-right = IMAGE-LEFT (small x).
    Uses connected components; falls back to a vertical midline split if the two
    lungs merge into one blob."""
    n, lbl = cv2.connectedComponents(mask.astype(np.uint8))
    comps = [(lbl == i) for i in range(1, n)]
    comps = [c for c in comps if c.sum() > 200]  # drop specks
    if len(comps) >= 2:
        comps.sort(key=lambda c: np.argwhere(c)[:, 1].mean())  # by mean x
        right = comps[0]
        left = np.logical_or.reduce(comps[1:])
    else:
        xs = np.argwhere(mask)[:, 1]
        mid = int(np.median(xs)) if xs.size else IMG // 2
        right = mask.copy(); right[:, mid:] = False
        left = mask.copy(); left[:, :mid] = False
    return right, left


def placement_region(mask: np.ndarray, left_upper_frac: float = 0.5,
                     erode_px: int = 6) -> np.ndarray:
    """Entire right lung ∪ upper `left_upper_frac` of the left lung, then eroded so
    lesions sit interior (off the mask edge and any residual cardiac bleed)."""
    right, left = split_lungs(mask)
    allowed = right.copy()
    ys = np.argwhere(left)
    if ys.size:
        ymin, ymax = int(ys[:, 0].min()), int(ys[:, 0].max())
        cut = ymin + int(left_upper_frac * (ymax - ymin))
        upper_left = left.copy()
        upper_left[cut:, :] = False
        allowed |= upper_left
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        allowed = cv2.erode(allowed.astype(np.uint8), k).astype(bool)
    return allowed


def _valid_centers(region: np.ndarray, lw: int, lh: int) -> np.ndarray:
    """Centres where a lw×lh box fits fully inside `region` (region eroded by box)."""
    kw, kh = lw | 1, lh | 1  # odd → centered anchor
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    return cv2.erode(region.astype(np.uint8), k).astype(bool)


def sample_center(region: np.ndarray, lw: int, lh: int,
                  rng: random.Random) -> tuple[int, int] | None:
    pts = np.argwhere(_valid_centers(region, lw, lh))  # (y,x)
    if pts.size == 0:
        return None
    y, x = pts[rng.randrange(len(pts))]
    return int(x), int(y)


# ── compositing ───────────────────────────────────────────────────────────────

def _ellipse_matte(lh: int, lw: int, scale: float) -> np.ndarray:
    m = np.zeros((lh, lw), np.uint8)
    ax = max(1, int(lw * scale / 2))
    ay = max(1, int(lh * scale / 2))
    cv2.ellipse(m, (lw // 2, lh // 2), (ax, ay), 0, 0, 360, 255, -1)
    return m


def _fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill interior holes — a cavitary lesion's dark core must stay INSIDE the matte."""
    h, w = m.shape
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)          # flood the outside
    return m | cv2.bitwise_not(ff)                 # anything unreached = interior hole


def _central_component(m: np.ndarray) -> np.ndarray:
    """Keep the blob nearest the box centre (the radiologist boxed ONE lesion)."""
    n, lbl = cv2.connectedComponents((m > 0).astype(np.uint8))
    if n <= 2:
        return m
    cy, cx = m.shape[0] / 2, m.shape[1] / 2
    best, bestd = None, None
    for i in range(1, n):
        pts = np.argwhere(lbl == i)
        d = ((pts[:, 0].mean() - cy) ** 2 + (pts[:, 1].mean() - cx) ** 2) / (pts.shape[0] ** 0.5)
        if bestd is None or d < bestd:
            best, bestd = i, d
    return np.where(lbl == best, 255, 0).astype(np.uint8)


def lesion_matte(crop: np.ndarray, method: str = "ellipse", scale: float = 0.9,
                 min_frac: float = 0.08, max_frac: float = 0.92) -> np.ndarray:
    """Matte (uint8 0/255) over the lesion inside its donor box.

    method="ellipse" is v1: a fixed ellipse at `scale` x the box — carries a lot of
    surrounding anatomy, and (worse) makes the inpainted background estimate garbage
    because the ring it interpolates from is far from the lesion. See README root cause.

    "otsu"/"dog" cut to the lesion itself. Both threshold |crop - smooth(crop)|, i.e.
    deviation of EITHER SIGN — 80.8% of real TB boxes here are DARKER at the core than
    the rim (cavitation), so a brightness threshold would segment the wrong thing.
    Holes are filled so a dark core stays inside the matte, and only the central blob
    is kept. Falls back to the ellipse if the result is degenerate (a failed Otsu on a
    low-contrast crop would otherwise silently produce an empty or box-filling matte).
    """
    lh, lw = crop.shape[:2]
    if method == "ellipse":
        return _ellipse_matte(lh, lw, scale)

    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(2.0, 0.25 * min(lh, lw))
    smooth = cv2.GaussianBlur(g, (0, 0), sigma)
    if method == "dog":
        dev = np.abs(cv2.GaussianBlur(g, (0, 0), max(1.0, sigma * 0.25)) - smooth)
    elif method == "otsu":
        dev = np.abs(g - smooth)
    else:
        raise ValueError(f"unknown matte method: {method}")

    d8 = cv2.normalize(dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, lw // 12) | 1,
                                                      max(3, lh // 12) | 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = _fill_holes(m)
    m = _central_component(m)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    frac = float((m > 0).mean())
    if not (min_frac <= frac <= max_frac):
        return _ellipse_matte(lh, lw, scale)      # degenerate → v1 geometry
    return m


def _new_box(center: tuple[int, int], lw: int, lh: int) -> tuple[int, int, int, int]:
    cx, cy = center
    x1, y1 = cx - lw // 2, cy - lh // 2
    clip = lambda v: int(np.clip(v, 0, IMG))
    return clip(x1), clip(y1), clip(x1 + lw), clip(y1 + lh)


def lesion_delta(crop: np.ndarray, matte: np.ndarray,
                 additive_only: bool = True, strength: float = 1.0) -> np.ndarray:
    """The lesion's density relative to the lung it sat on: `crop - inpaint(crop)`.

    Physically this is the right quantity — pasting it onto a new lung reproduces the
    lesion's own contrast on the target's anatomy, and the donor's ribs cancel because
    they are present in BOTH crop and background.

    `additive_only` is the v1 behaviour and it is WRONG for this dataset (kept as the
    default only so v1 stays reproducible): 80.8% of real TB boxes are darker at the
    core than the rim, and the inpainted background comes out BRIGHTER than the lesion
    (131.6 vs 113.9 measured), so clipping to >=0 discards the lesion and keeps +5.94
    grey of rim against a real ~20-grey contrast. Pass False with a TIGHT matte.
    """
    bg = cv2.inpaint(crop, matte, 5, cv2.INPAINT_TELEA).astype(np.float32)
    d = crop.astype(np.float32) - bg
    if additive_only:
        d = np.clip(d, 0, None)
    return d * strength


def _residual_paste(canvas: np.ndarray, crop: np.ndarray, matte: np.ndarray,
                    center: tuple[int, int], strength: float = 1.0,
                    additive_only: bool = True) -> np.ndarray:
    """Blend the lesion's density onto the target, feathered so the edge fades."""
    lh, lw = crop.shape[:2]
    delta = lesion_delta(crop, matte, additive_only, strength)
    alpha = cv2.GaussianBlur(matte.astype(np.float32) / 255.0, (0, 0),
                             max(1.0, lw * 0.06))[..., None]  # soft edge
    cx, cy = center
    x0, y0 = cx - lw // 2, cy - lh // 2
    out = canvas.astype(np.float32).copy()
    reg = out[y0:y0 + lh, x0:x0 + lw]
    out[y0:y0 + lh, x0:x0 + lw] = np.clip(reg + delta * alpha, 0, 255)
    return out.astype(np.uint8)


def paste(canvas: np.ndarray, lesion: Lesion, center: tuple[int, int],
          matte_scale: float = 0.9, blend: str = "mixed", strength: float = 1.0,
          resize_to: tuple[int, int] | None = None,
          matte_method: str = "ellipse", additive_only: bool = True,
          return_matte: bool = False):
    """Blend `lesion` into `canvas` (RGB uint8) centred at `center`. Returns
    (out_rgb, new_box_px), or (out_rgb, box, matte) with `return_matte`.

    blend ∈ {"mixed" (MIXED_CLONE), "normal" (NORMAL_CLONE), "residual"}.
      * MIXED_CLONE keeps whichever gradient is stronger — with a TIGHT matte that can
        let the target's ribs overwrite the lesion's own structure.
      * NORMAL_CLONE imports every source gradient in the matte. That was unusable at
        the v1 ellipse (it dragged in donor ribs), but with a tight matte the only
        gradients inside are the lesion's — and Poisson solves the DC offset from the
        target boundary, preserving dark-core/bright-rim structure.
      * "residual" adds `lesion_delta` (see there re: `additive_only`).

    `matte_method` ∈ {"ellipse" (v1), "otsu", "dog"}; `additive_only` only affects the
    residual blend. Defaults reproduce v1 exactly."""
    if resize_to is not None:
        lw, lh = resize_to
        crop = cv2.resize(lesion.crop, (lw, lh), interpolation=cv2.INTER_AREA)
    else:
        x1, y1, x2, y2 = lesion.box
        lw, lh = x2 - x1, y2 - y1
        crop = lesion.crop
    matte = lesion_matte(crop, matte_method, matte_scale)
    if blend == "residual":
        out = _residual_paste(canvas, crop, matte, center, strength, additive_only)
    else:
        mode = cv2.NORMAL_CLONE if blend == "normal" else cv2.MIXED_CLONE
        # cv2.seamlessClone MUTATES its mask in place (verified: sum halves). Hand it a
        # copy so callers get back the matte we actually built.
        out = cv2.seamlessClone(crop, canvas, matte.copy(), center, mode)
    box = _new_box(center, lw, lh)
    return (out, box, matte) if return_matte else (out, box)


def fit_placement(region: np.ndarray, lesion: Lesion, rng: random.Random,
                  min_scale: float = 0.35, step: float = 0.1,
                  allow_downscale: bool = True):
    """A centre + (w,h) for the lesion in `region`, downscaling ONLY if it won't fit
    at full size. Returns (center, (w,h), scale) or None. scale=1.0 = unchanged;
    <1.0 means the lesion was too big for the safe region and got shrunk to fit.

    `allow_downscale=False` skips a lesion that doesn't fit rather than shrinking it:
    INTER_AREA resampling leaves the patch locally smoother than its surroundings, a
    cue the detector can pick up (39% of the v1 pool was downscaled, some to 0.4x).
    Costs pool size — measure the trade."""
    lw0 = lesion.box[2] - lesion.box[0]
    lh0 = lesion.box[3] - lesion.box[1]
    if not allow_downscale:
        min_scale = 1.0
    scale = 1.0
    while scale >= min_scale - 1e-9:
        w = max(6, int(round(lw0 * scale)))
        h = max(6, int(round(lh0 * scale)))
        c = sample_center(region, w, h, rng)
        if c is not None:
            return c, (w, h), round(scale, 3)
        scale -= step
    return None


def choose_placement(lung_mask: np.ndarray, bank: list[Lesion], rng: random.Random,
                     left_upper_frac: float = 0.5, erode_px: int = 6, tries: int = 25):
    """Pick one random lesion + a fitting centre in the safe region. Returns
    (lesion, center, region) or None. Blend is applied separately (via `paste`) so
    the SAME placement can be rendered with different blend modes for comparison."""
    region = placement_region(lung_mask, left_upper_frac, erode_px)
    for _ in range(tries):
        les = bank[rng.randrange(len(bank))]
        lw = les.box[2] - les.box[0]
        lh = les.box[3] - les.box[1]
        c = sample_center(region, lw, lh, rng)
        if c is not None:
            return les, c, region
    return None


def yolo_label_lines(labels: list[tuple[int, tuple[int, int, int, int]]],
                     size: int = IMG) -> str:
    """[(cls,(x1,y1,x2,y2))] → normalized YOLO '<cls> cx cy w h' lines."""
    out = []
    for cls, (x1, y1, x2, y2) in labels:
        cx = (x1 + x2) / 2 / size
        cy = (y1 + y2) / 2 / size
        w = (x2 - x1) / size
        h = (y2 - y1) / size
        out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def fake_lung_mask() -> np.ndarray:
    """Two-ellipse stand-in lung mask for plumbing tests without the segmenter.
    Left ellipse = image-LEFT = patient-right; right ellipse = patient-left."""
    yy, xx = np.ogrid[:IMG, :IMG]
    m = np.zeros((IMG, IMG), bool)
    for cx in (170, 342):
        m |= (((xx - cx) ** 2) / (72 ** 2) + ((yy - 260) ** 2) / (150 ** 2)) <= 1
    return m
