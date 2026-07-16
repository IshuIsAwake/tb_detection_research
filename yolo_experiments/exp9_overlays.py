"""
exp9_overlays.py — Box-quality eyeball for the cross-domain detector.

The external Montgomery/Shenzhen sets have NO lesion boxes, so we can't compute
box mAP. Instead we (a) draw the detector's boxes on the images for manual
review, and (b) back the eyeball with two cheap semi-quantitative checks that
DON'T need pixel-accurate GT:

  1. Quadrant agreement — parse the coarse lesion location from the clinical
     reading text ("right upper lobe" → a quadrant), draw it as a RED box, and
     check whether any detector box (BLUE) lands in that quadrant.
     NOTE: chest X-ray laterality is from the PATIENT'S view — patient-right is
     on the IMAGE-LEFT. The red box mirrors accordingly.
  2. In-lung fraction (Montgomery only) — using the manual lung masks, what
     fraction of each detector box sits inside the lung field? A spray detector
     would score low; a real one high.

Uses the cached predictions.json from exp9_cross_domain.py (no re-inference).
Default config = best_512_tta (the cross-domain winner), seed 0.

    python yolo_experiments/exp9_overlays.py
    python yolo_experiments/exp9_overlays.py --config best_512 --conf 0.25
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import settings as S

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

EXTRA = S.IMGS_ROOT / "extra" / "mc+shenzhen"
EXT = S.REPO_ROOT / "data_external"
RUN = S.RESULTS_ROOT / "exp9_cross_domain"
IMG = 512


def find_image(name: str) -> Path | None:
    for p in EXTRA.rglob(f"{name}.png"):
        return p
    return None


def reading_text(source: str, name: str) -> str:
    d = "montgomery" if source == "Montgomery" else "shenzhen"
    f = EXT / d / "ClinicalReadings" / f"{name}.txt"
    if not f.exists():
        return ""
    return f.read_text(errors="ignore").strip()


# ── location text → coarse quadrant box (512-space), patient→image mirrored ────
# Vocabulary is messy clinical shorthand: lobe codes RUL/LUL/RML/RLL/LLL/LML,
# words (upper/apical/apex, lower/base, middle/mid-zone), laterality
# (right/rt/R, left/lt/L), and whole-lung terms (bilateral/both/diffuse/miliary).
_ZONE_Y = {"upper": (0.00, 0.50), "middle": (0.30, 0.70), "lower": (0.50, 1.00)}


def quadrant_box(text: str):
    """Return (x1,y1,x2,y2) in 512-space for the union of described regions, or
    None if the reading carries no localisation info."""
    t = text.lower()
    zones, sides = set(), set()

    # lobe codes:  r/l  +  u/m/l  + l   → RUL, LLL, RML, ...
    for side, z in re.findall(r"\b([rl])(u|m|l)l\b", t):
        sides.add("R" if side == "r" else "L")
        zones.add({"u": "upper", "m": "middle", "l": "lower"}[z])

    # worded verticals
    if re.search(r"upper|apical|apex|apic", t):            zones.add("upper")
    if re.search(r"lower|base|basal|basilar", t):          zones.add("lower")
    if re.search(r"middle|mid[\s-]?zone|mid[\s-]?lung", t): zones.add("middle")

    # worded laterality
    if re.search(r"\bright\b|\brt\b|\br\.", t):  sides.add("R")
    if re.search(r"\bleft\b|\blt\b|\bl\.|lingula", t):  sides.add("L")

    bilateral = bool(re.search(r"bilateral|both lung|diffuse|widespread|miliary", t))

    if not zones and not sides and not bilateral:
        return None

    # vertical span = union of zone bands (unknown → full height)
    if zones:
        tops = [_ZONE_Y[z][0] for z in zones]; bots = [_ZONE_Y[z][1] for z in zones]
        y1, y2 = min(tops), max(bots)
    else:
        y1, y2 = 0.0, 1.0
    # horizontal: PATIENT side mirrored to IMAGE side (patient-R → image-LEFT)
    if bilateral or {"R", "L"} <= sides:
        x1, x2 = 0.0, 1.0
    elif sides == {"R"}:
        x1, x2 = 0.0, 0.5
    elif sides == {"L"}:
        x1, x2 = 0.5, 1.0
    else:
        x1, x2 = 0.0, 1.0
    if (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0):
        return None
    return (x1 * IMG, y1 * IMG, x2 * IMG, y2 * IMG)


def load_lung_mask(name: str):
    """Union of L+R Montgomery masks, resized to 512x512 (bool), or None."""
    l = EXT / "montgomery" / "leftMask" / f"{name}.png"
    r = EXT / "montgomery" / "rightMask" / f"{name}.png"
    if not (l.exists() and r.exists()):
        return None
    ml = np.array(Image.open(l).convert("L").resize((IMG, IMG))) > 127
    mr = np.array(Image.open(r).convert("L").resize((IMG, IMG))) > 127
    return ml | mr


def box_center_in(box, region):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def in_lung_fraction(box, mask):
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMG, x2), min(IMG, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    return float(sub.mean()) if sub.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="best_512_tta")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--per-page", type=int, default=20)
    args = ap.parse_args()

    preds_all = json.loads((RUN / "predictions.json").read_text())
    preds = preds_all[args.config][args.seed]

    out = RUN / "overlays" / args.config
    out.mkdir(parents=True, exist_ok=True)

    stats = {"Montgomery": {"n": 0, "quad_n": 0, "quad_hit": 0,
                            "inlung_boxes": 0, "inlung_sum": 0.0},
             "Shenzhen": {"n": 0, "quad_n": 0, "quad_hit": 0}}

    # gather TB images per source
    for source in ("Montgomery", "Shenzhen"):
        tb = [(k, v) for k, v in preds.items()
              if v["label"] == "tb" and v["source"] == source]
        tb.sort()
        panels = []
        for name, rec in tb:
            imgp = find_image(name)
            if imgp is None:
                continue
            txt = reading_text(source, name)
            quad = quadrant_box(txt)
            boxes = [b for b in rec["boxes"] if b[4] >= args.conf]
            mask = load_lung_mask(name) if source == "Montgomery" else None

            st = stats[source]
            st["n"] += 1
            if quad is not None:
                st["quad_n"] += 1
                if any(box_center_in(b, quad) for b in boxes):
                    st["quad_hit"] += 1
            if mask is not None:
                for b in boxes:
                    st["inlung_boxes"] += 1
                    st["inlung_sum"] += in_lung_fraction(b, mask)
            panels.append((name, imgp, txt, quad, boxes, mask))

        # render montage pages
        pp = args.per_page
        for pg in range((len(panels) + pp - 1) // pp):
            chunk = panels[pg * pp:(pg + 1) * pp]
            cols = 5
            rows = (len(chunk) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
            axes = np.array(axes).reshape(-1)
            for ax in axes:
                ax.axis("off")
            for ax, (name, imgp, txt, quad, boxes, mask) in zip(axes, chunk):
                ax.imshow(Image.open(imgp).convert("L"), cmap="gray")
                if mask is not None:
                    ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.6)
                if quad is not None:
                    ax.add_patch(Rectangle((quad[0], quad[1]), quad[2]-quad[0],
                                 quad[3]-quad[1], fill=False, edgecolor="red",
                                 lw=1.5, linestyle="--"))
                for b in boxes:
                    ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                 fill=False, edgecolor="deepskyblue", lw=1.3))
                    ax.text(b[0], b[1]-2, f"{b[4]:.2f}", color="deepskyblue",
                            fontsize=6, va="bottom")
                short = txt.splitlines()[-1][:38] if txt else "(no reading)"
                ax.set_title(f"{name}\n{short}", fontsize=6)
            fig.suptitle(f"{source} TB — {args.config} s{args.seed} "
                         f"(blue=detector≥{args.conf}, red=reading quadrant, "
                         f"green=lung)", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fp = out / f"{source.lower()}_tb_p{pg+1}.png"
            fig.savefig(fp, dpi=110)
            plt.close(fig)
            print(f"→ {fp}")

    # summary
    print("\n── box-quality summary ──")
    summ = {}
    for source, st in stats.items():
        qa = (st["quad_hit"] / st["quad_n"]) if st["quad_n"] else None
        line = {"tb_images": st["n"],
                "quadrant_localised": st["quad_n"],
                "quadrant_agreement": round(qa, 3) if qa is not None else None}
        if source == "Montgomery" and st["inlung_boxes"]:
            line["mean_in_lung_fraction"] = round(st["inlung_sum"] / st["inlung_boxes"], 3)
            line["n_boxes"] = st["inlung_boxes"]
        summ[source] = line
        print(f"  {source}: {line}")
    (out / "box_quality.json").write_text(json.dumps(
        {"config": args.config, "seed": args.seed, "conf": args.conf,
         "summary": summ}, indent=2))
    print(f"\n→ {out}/box_quality.json")


if __name__ == "__main__":
    main()
