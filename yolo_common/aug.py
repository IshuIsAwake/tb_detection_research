"""
aug.py — Named augmentation levels (Ultralytics train kwargs).

exp1 is a deliberate, unoptimised FLOOR: it uses "off" — every augmentation
explicitly zeroed. Ultralytics augments by default (mosaic, fliplr, hsv, scale,
translate, erasing...), so "off" must be spelled out, not left implicit.

Later experiments select other levels to sweep augmentation as the single
controlled variable against the same frozen split.
"""

from __future__ import annotations

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
    # Mild geometry, colour locked (X-ray intensity is diagnostic).
    "light": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 5.0, "translate": 0.05, "scale": 0.3,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "default": {},   # Ultralytics defaults — for an explicit "what default aug does" arm
}


def describe(level: str) -> str:
    cfg = AUG_LEVELS[level]
    if not cfg:
        return f"{level}: Ultralytics defaults (no overrides)"
    return f"{level}: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
