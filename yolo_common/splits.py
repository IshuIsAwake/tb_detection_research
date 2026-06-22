"""
splits.py — The frozen split, built once and never regenerated.

`splits.json` is the permanent reference for EVERY future experiment. Build it
once; on every later call it is loaded, never rebuilt (the on-disk file is the
authority). Deterministic from SEED, atomic write.

Layout:
  positives = the ~799 TB images that carry boxes (non-empty labels).
  Stratify by class signature {active-only, obsolete-only, both}, then deal
  70/15/15 → train / val / test_positives  (≈ 559 / 120 / 120).

  Sealed test set (the permanent blackbox) =
      test_positives (TB)  +  120 sick  +  120 healthy
  the negatives sampled fixed-seed from their folders.

╔══════════════════════════════════════════════════════════════════════════╗
║ CRITICAL LEAKAGE RULE                                                      ║
║ `blackbox_negative_ids` (the 120 sick + 120 healthy) are RESERVED. Every  ║
║ future experiment that adds negatives to TRAINING must draw from the      ║
║ remaining ~3680 sick and ~3680 healthy and MUST exclude these 240.        ║
║ Otherwise a future run trains on images sitting in the final test set and ║
║ silently breaks the blackbox. This rule is also recorded inside           ║
║ splits.json itself.                                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

`materialise()` builds the YOLO train/val tree (`yolo_datasets/det/`) with
symlinks to the read-only originals + the converted labels, plus `data.yaml`.
Same images for both 512 and 1024 runs — imgsz handles resize, no per-res copy.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from yolo_common import convert, settings as S

LEAKAGE_RULE = (
    "blackbox_negative_ids are RESERVED. Any future experiment adding negatives "
    "to TRAINING must exclude these 240 ids (draw from the other ~3680 sick / "
    "~3680 healthy). Reusing them in training leaks the sealed test set."
)


def _stratified_split(stems_by_sig: dict[str, list[str]], seed: int):
    """Deal each signature bucket 70/15/15 → (train, val, test) stem lists."""
    train, val, test = [], [], []
    for sig in sorted(stems_by_sig):
        bucket = sorted(stems_by_sig[sig])
        random.Random(seed * 31 + hash(sig) % 997).shuffle(bucket)
        n = len(bucket)
        n_tr = round(n * S.SPLIT_TRAIN_FRAC)
        n_va = round(n * S.SPLIT_VAL_FRAC)
        train += bucket[:n_tr]
        val += bucket[n_tr:n_tr + n_va]
        test += bucket[n_tr + n_va:]
    return sorted(train), sorted(val), sorted(test)


def build_or_load() -> dict:
    """Build splits.json once (atomic); load it on every subsequent call."""
    if S.SPLITS_JSON.exists():
        return json.loads(S.SPLITS_JSON.read_text())

    convert.build_or_load()                      # ensure labels + signatures exist
    sigs = convert.class_signatures()            # stem → active/obsolete/both
    by_sig: dict[str, list[str]] = defaultdict(list)
    for stem, sig in sigs.items():
        by_sig[sig].append(stem)

    train_ids, val_ids, test_pos_ids = _stratified_split(by_sig, S.SEED)

    # Fixed-seed negative sampling for the sealed test set.
    rng = random.Random(S.SEED + 17)
    sick = [p.stem for p in S.list_images(S.SICK_IMAGES)]
    healthy = [p.stem for p in S.list_images(S.HEALTH_IMAGES)]
    rng.shuffle(sick)
    rng.shuffle(healthy)
    bb_sick = sorted(sick[:S.N_BLACKBOX_NEG_PER_GROUP])
    bb_healthy = sorted(healthy[:S.N_BLACKBOX_NEG_PER_GROUP])

    out = {
        "seed": S.SEED,
        "leakage_rule": LEAKAGE_RULE,
        "fractions": {"train": S.SPLIT_TRAIN_FRAC, "val": S.SPLIT_VAL_FRAC,
                      "test_positives": S.SPLIT_TEST_FRAC},
        "class_map": {str(v): S.CLASS_NAMES[v] for v in S.COCO_CAT_TO_YOLO.values()},
        "counts": {
            "train_positives": len(train_ids),
            "val_positives": len(val_ids),
            "test_positives": len(test_pos_ids),
            "blackbox_sick": len(bb_sick),
            "blackbox_healthy": len(bb_healthy),
        },
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_positive_ids": test_pos_ids,
        "blackbox_negative_ids": {"sick": bb_sick, "healthy": bb_healthy},
    }
    S.SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = S.SPLITS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(S.SPLITS_JSON)                    # atomic
    return out


# ── path helpers ──────────────────────────────────────────────────────────────

def tb_image_path(stem: str) -> Path:
    return S.TB_IMAGES / f"{stem}.png"


def label_path(stem: str) -> Path:
    return S.LABELS_ALL_DIR / f"{stem}.txt"


def negative_image_path(stem: str) -> Path:
    """h* → health/, s* → sick/."""
    if stem.startswith("h"):
        return S.HEALTH_IMAGES / f"{stem}.png"
    if stem.startswith("s"):
        return S.SICK_IMAGES / f"{stem}.png"
    raise ValueError(f"unknown negative stem {stem!r}")


def select_train_negatives(n_total: int, sick_frac: float = 0.5,
                           seed: int = None, split: dict | None = None) -> dict:
    """Pick training-negative stems, fixed-seed, EXCLUDING the 240 reserved
    blackbox negatives. Returns {'sick': [...], 'healthy': [...]}.

    Leakage-safe by construction: the reserved ids never enter training.
    """
    split = split or build_or_load()
    seed = S.SEED if seed is None else seed
    reserved = set(split["blackbox_negative_ids"]["sick"]) \
        | set(split["blackbox_negative_ids"]["healthy"])

    sick_pool = [p.stem for p in S.list_images(S.SICK_IMAGES) if p.stem not in reserved]
    healthy_pool = [p.stem for p in S.list_images(S.HEALTH_IMAGES) if p.stem not in reserved]
    random.Random(seed + 101).shuffle(sick_pool)
    random.Random(seed + 202).shuffle(healthy_pool)

    n_sick = round(n_total * sick_frac)
    n_healthy = n_total - n_sick
    return {"sick": sorted(sick_pool[:n_sick]),
            "healthy": sorted(healthy_pool[:n_healthy])}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def materialise(tree: Path | None = None,
                train_negative_stems: list[str] | None = None,
                smoke: bool = False) -> Path:
    """Build a YOLO tree {images,labels}/{train,val} + data.yaml under `tree`
    (default yolo_datasets/det/). Returns the data.yaml path.

    Ultralytics infers a label path by swapping the 'images' segment for
    'labels' and the ext for '.txt', so the parallel tree must use exactly
    those dir names.

    `train_negative_stems` (exp2+) adds background images to TRAIN only, each
    with an EMPTY label file (Ultralytics treats that as a true negative).
    val is left positives-only so best-weight selection stays comparable to
    exp1 — the false-alarm effect shows up in the sealed blackbox eval.
    """
    split = build_or_load()
    convert.build_or_load()
    tree = tree or S.DET_TREE

    train_ids = list(split["train_ids"])
    val_ids = list(split["val_ids"])
    neg = list(train_negative_stems or [])
    if smoke:
        train_ids = train_ids[:S.SMOKE_SUBSET]
        val_ids = val_ids[:max(2, S.SMOKE_SUBSET // 4)]
        neg = neg[:S.SMOKE_SUBSET]

    # Clear stale links first so the set is deterministic regardless of a prior
    # (e.g. full vs smoke) materialise. Symlinks/empty files — cheap to rebuild.
    import shutil
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = tree / sub
        if d.exists():
            shutil.rmtree(d)

    for subset, ids in (("train", train_ids), ("val", val_ids)):
        for stem in ids:
            _link(tb_image_path(stem), tree / "images" / subset / f"{stem}.png")
            _link(label_path(stem), tree / "labels" / subset / f"{stem}.txt")

    for stem in neg:                                  # background negatives → train
        _link(negative_image_path(stem), tree / "images" / "train" / f"{stem}.png")
        lbl = tree / "labels" / "train" / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("")                            # empty = background

    yaml_path = tree / "data.yaml"
    yaml_path.write_text(
        f"# generated by yolo_common/splits.py — rebuildable, do not hand-edit\n"
        f"path: {tree.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {S.NUM_CLASSES}\n"
        f"names: {S.CLASS_NAMES}\n"
    )
    return yaml_path


def blackbox(split: dict | None = None) -> list[tuple[Path, Path | None, str]]:
    """The sealed 360: list of (image_path, label_path_or_None, group).

    group ∈ {'tb','sick','healthy'}. Negatives have label None (empty GT).
    """
    split = split or build_or_load()
    items: list[tuple[Path, Path | None, str]] = []
    for stem in split["test_positive_ids"]:
        items.append((tb_image_path(stem), label_path(stem), "tb"))
    for stem in split["blackbox_negative_ids"]["sick"]:
        items.append((negative_image_path(stem), None, "sick"))
    for stem in split["blackbox_negative_ids"]["healthy"]:
        items.append((negative_image_path(stem), None, "healthy"))
    return items


if __name__ == "__main__":
    s = build_or_load()
    print(json.dumps(s["counts"], indent=2))
