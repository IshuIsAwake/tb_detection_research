"""
convert.py — TBX11K COCO JSON → YOLO labels (the prep step).

The TBX11K data ships as COCO JSON + VOC XML, NOT YOLO. This module is the one
place that turns the COCO annotations into YOLO `.txt` label files. It is
deliberately a separate, importable prep step (not embedded in any experiment)
so every experiment shares byte-identical labels.

Source of truth: `TBX11K_trainval_only_tb.json` — 512x512 coords, one entry per
TB-positive image (799 of the 800 tb/ images; one tb image carries no box and
is therefore not a detection positive). COCO bbox is [x, y, w, h] in pixels.

YOLO line per box:  `<cls> <cx> <cy> <w> <h>`  (class id + normalised centre/size)
Class map (verified against the JSON, not assumed):
    COCO 1 ActiveTuberculosis            → YOLO 0
    COCO 2 ObsoletePulmonaryTuberculosis → YOLO 1
    COCO 3 PulmonaryTuberculosis         → (absent from trainval boxes)

build_or_load: writes `yolo_datasets/labels_all/<stem>.txt` once and a
`convert_report.json`. Idempotent — re-run is a no-op load. Safe to delete the
whole `yolo_datasets/` tree and rebuild.

CLI:  python yolo_common/convert.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# path shim so `python yolo_common/convert.py` works standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import settings as S


def load_coco(json_path: Path = S.ANN_JSON) -> dict:
    return json.loads(Path(json_path).read_text())


def _verify_class_map(coco: dict) -> None:
    """Halt if the COCO category ids/names don't match our hard-coded map."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    for cat_id, yolo_id in S.COCO_CAT_TO_YOLO.items():
        expect = S.CLASS_NAMES[yolo_id]
        got = cats.get(cat_id)
        if got != expect:
            raise SystemExit(
                f"[convert] CLASS MAP CONTRADICTION: COCO cat {cat_id} is "
                f"{got!r} but settings expects {expect!r}. Halting — do not "
                f"silently 'fix'. Inspect {S.ANN_JSON}."
            )


def _coco_index(coco: dict):
    """Return (img_by_id, anns_by_img). file_name like 'tb/tb0003.png'."""
    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    return img_by_id, anns_by_img


def _yolo_lines(anns: list, w: int, h: int) -> list[str]:
    lines = []
    for a in anns:
        cid = a["category_id"]
        if cid not in S.COCO_CAT_TO_YOLO:
            continue  # cat 3 etc. — absent in trainval, skip defensively
        x, y, bw, bh = a["bbox"]
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        lines.append(f"{S.COCO_CAT_TO_YOLO[cid]} {cx:.6f} {cy:.6f} "
                     f"{bw / w:.6f} {bh / h:.6f}")
    return lines


def stem_of(file_name: str) -> str:
    """'tb/tb0003.png' → 'tb0003'."""
    return Path(file_name).stem


def class_signatures(coco: dict | None = None) -> dict[str, str]:
    """stem → one of {'active', 'obsolete', 'both'} for stratified splitting."""
    coco = coco or load_coco()
    _, anns_by_img = _coco_index(coco)
    sig: dict[str, str] = {}
    for im in coco["images"]:
        ys = {S.COCO_CAT_TO_YOLO[a["category_id"]]
              for a in anns_by_img[im["id"]]
              if a["category_id"] in S.COCO_CAT_TO_YOLO}
        if not ys:
            continue
        sig[stem_of(im["file_name"])] = (
            "both" if ys == {0, 1} else ("active" if ys == {0} else "obsolete")
        )
    return sig


def build_or_load(force: bool = False) -> dict:
    """Build labels_all/<stem>.txt + convert_report.json once; else load report."""
    if S.CONVERT_REPORT.exists() and not force:
        return json.loads(S.CONVERT_REPORT.read_text())

    coco = load_coco()
    _verify_class_map(coco)
    img_by_id, anns_by_img = _coco_index(coco)

    S.LABELS_ALL_DIR.mkdir(parents=True, exist_ok=True)
    box_hist: Counter = Counter()
    sig_hist: Counter = Counter()
    n_imgs = 0
    sigs = class_signatures(coco)

    for im in coco["images"]:
        w, h = im["width"], im["height"]
        anns = anns_by_img[im["id"]]
        lines = _yolo_lines(anns, w, h)
        if not lines:
            continue
        stem = stem_of(im["file_name"])
        (S.LABELS_ALL_DIR / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_imgs += 1
        for ln in lines:
            box_hist[int(ln.split()[0])] += 1
        sig_hist[sigs.get(stem, "?")] += 1

    report = {
        "source_json": str(S.ANN_JSON),
        "labels_dir": str(S.LABELS_ALL_DIR),
        "n_positive_images": n_imgs,
        "box_counts": {S.CLASS_NAMES[k]: v for k, v in sorted(box_hist.items())},
        "total_boxes": sum(box_hist.values()),
        "class_signature_counts": dict(sig_hist),
        "class_map": {str(v): S.CLASS_NAMES[v] for v in S.COCO_CAT_TO_YOLO.values()},
    }
    S.CONVERT_REPORT.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = build_or_load(force="--force" in sys.argv)
    print(json.dumps(rep, indent=2))
