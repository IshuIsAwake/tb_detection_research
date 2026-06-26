"""
aug.py — Named augmentation levels (Ultralytics train kwargs).

exp1 is a deliberate, unoptimised FLOOR: it uses "off" — every augmentation
explicitly zeroed. Ultralytics augments by default (mosaic, fliplr, hsv, scale,
translate, erasing...), so "off" must be spelled out, not left implicit.

Later experiments select other levels to sweep augmentation as the single
controlled variable against the same frozen split.
"""

from __future__ import annotations

# Realistic geometry for a frontal chest film: small rotation + translation
# (patient positioning), scale (body size / detector distance — and our most
# direct lever on the small/medium-lesion recall wall), horizontal flip (lesions
# are bilateral and the detector keys on lung texture, not cardiac laterality).
# flipud / shear / perspective stay OFF — a chest X-ray has a fixed upright,
# near-orthographic pose, so they only add label noise. hsv_h / hsv_s are no-ops
# on grayscale (R=G=B → saturation 0); only hsv_v (brightness) has any effect.
_GEO: dict = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 10.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
    "erasing": 0.0,
}

AUG_LEVELS: dict[str, dict] = {
    # True no-aug control — the exp1 floor. Everything off.
    "off": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "erasing": 0.0,
    },
    # Mild geometry, colour locked (X-ray intensity is diagnostic). Kept for the
    # record; exp3 escalates from `geo` (stronger scale) upward.
    "light": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 5.0, "translate": 0.05, "scale": 0.3,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "erasing": 0.0,
    },
    # ── exp3 escalation (positives-only screen, 512 @ batch 16) ──────────────
    # Realistic geometry only.
    "geo": dict(_GEO),
    # + exposure jitter (the only physically-grounded photometric knob here).
    "geo_photo": {**_GEO, "hsv_v": 0.4},
    # + YOLO's strongest regulariser: mosaic juxtaposes 4 (scaled) chests.
    # close_mosaic disables it for the last epochs so training ends on real
    # images. Tests whether synthetic composites beat the small-lesion / overfit
    # wall on CXR, or whether the anatomical nonsense hurts.
    "mosaic": {**_GEO, "mosaic": 1.0, "close_mosaic": 10},
    # Optional follow-up: add mixup (translucent double-exposure of two chests).
    # Run ONLY if the `mosaic` arm shows synthetic composites help.
    "mosaic_mixup": {**_GEO, "mosaic": 1.0, "mixup": 0.1, "close_mosaic": 10},
    # Ultralytics defaults — the kitchen-sink reference. With albumentations
    # installed this also pulls in Blur/CLAHE/RandomBrightnessContrast/etc.
    "default": {},
}


def describe(level: str) -> str:
    cfg = AUG_LEVELS[level]
    if not cfg:
        return f"{level}: Ultralytics defaults (no overrides)"
    return f"{level}: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
