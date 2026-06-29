"""
kfold.py — Stratified k-fold cross-validation over ALL TB positives.

WHY THIS EXISTS (and why it does NOT touch splits.json)
    splits.json is the FROZEN reference split for the fixed-split experiments
    (exp1-4, exp6). It must stay byte-identical so those runs stay reproducible.
    exp5 asks a different question — "does the finalist config survive WHICH
    images land in the test set?" — so it needs its own partition. This module
    builds that partition and never reads or writes splits.json.

WHAT IT BUILDS
    A k-way partition of the ~799 TB positives, STRATIFIED by class signature
    {active, obsolete, both} so every fold sees the same class mix. The test
    fold ROTATES: across the k runs every positive is tested exactly once.
    POSITIVES-ONLY — no negatives (false alarms are the classifier's job; see
    research-objective-method). For each outer fold, an inner-val slice is carved
    (stratified) from the k-1 training folds purely for best.pt selection.

    folds.json is written ONCE under yolo_datasets/kfold/ (rebuildable, NOT the
    frozen split) and loaded thereafter, so every per-fold invocation deals the
    exact same folds. Deterministic from (SEED, k); halts if an existing
    folds.json disagrees on (seed, k) rather than silently reusing a stale one.

Layout built per fold (Ultralytics infers labels by swapping images→labels):
    yolo_datasets/kfold/fold<i>/{images,labels}/{train,val}  + data.yaml
        train = (k-1 folds) minus inner-val      val = inner-val (best.pt only)
    Eval (the rotating test fold) runs separately via fold_eval_inputs().
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# path shim so `python yolo_common/kfold.py` works standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import convert, settings as S, splits

# Generated, rebuildable — NOT the frozen split. Safe to delete.
KFOLD_ROOT = S.GEN_ROOT / "kfold"
FOLDS_JSON = KFOLD_ROOT / "folds.json"


def _stratified_folds(stems_by_sig: dict[str, list[str]], k: int, seed: int) -> list[list[str]]:
    """Deal each signature bucket round-robin across k folds (balanced + stratified)."""
    folds: list[list[str]] = [[] for _ in range(k)]
    for sig in sorted(stems_by_sig):
        bucket = sorted(stems_by_sig[sig])
        random.Random(seed * 31 + hash(sig) % 997).shuffle(bucket)
        for i, stem in enumerate(bucket):
            folds[i % k].append(stem)
    return [sorted(f) for f in folds]


def build_or_load_folds(k: int, seed: int) -> dict:
    """Build folds.json once (atomic); load it on every subsequent call.

    Halts if an existing folds.json was built for a different (seed, k) — the
    per-fold runs must all share one partition, so a mismatch is a contradiction
    to surface, not paper over (delete yolo_datasets/kfold/ to rebuild)."""
    if FOLDS_JSON.exists():
        data = json.loads(FOLDS_JSON.read_text())
        if data["seed"] != seed or data["k"] != k:
            raise SystemExit(
                f"[kfold] {FOLDS_JSON} was built for seed={data['seed']} k={data['k']} "
                f"but this run asks seed={seed} k={k}. Folds must be shared across "
                f"all per-fold invocations. Delete {KFOLD_ROOT} to rebuild, or align "
                f"SEED/--k.")
        return data

    convert.build_or_load()                      # ensure labels + signatures exist
    sigs = convert.class_signatures()            # stem → active/obsolete/both (all 799)
    by_sig: dict[str, list[str]] = defaultdict(list)
    for stem, sig in sigs.items():
        by_sig[sig].append(stem)

    folds = _stratified_folds(by_sig, k, seed)
    # Per-fold signature histogram — proves the stratification actually balanced.
    sig_hist = [dict(sorted({s: sum(sigs[x] == s for x in f)
                             for s in set(sigs.values())}.items())) for f in folds]

    out = {
        "seed": seed,
        "k": k,
        "note": "k-fold partition over ALL TB positives; independent of the frozen "
                "splits.json. Test fold rotates; positives-only.",
        "n_positives": sum(len(f) for f in folds),
        "fold_sizes": [len(f) for f in folds],
        "fold_signatures": sig_hist,
        "folds": folds,
    }
    KFOLD_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = FOLDS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(FOLDS_JSON)                       # atomic
    return out


def _carve_inner_val(train_stems: list[str], sigs: dict[str, str],
                     frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out a stratified `frac` of the training pool for best.pt selection."""
    by_sig: dict[str, list[str]] = defaultdict(list)
    for s in train_stems:
        by_sig[sigs[s]].append(s)
    val: list[str] = []
    for sig in sorted(by_sig):
        bucket = sorted(by_sig[sig])
        random.Random(seed * 53 + hash(sig) % 997).shuffle(bucket)
        n_val = max(1, round(len(bucket) * frac))
        val += bucket[:n_val]
    val_set = set(val)
    train = [s for s in train_stems if s not in val_set]
    return sorted(train), sorted(val)


def fold_partition(folds_data: dict, fold_idx: int, *, inner_val_frac: float,
                   seed: int, smoke: bool = False) -> dict:
    """Return {'train','inner_val','test'} stem lists for one outer fold.

    test = fold `fold_idx`; train+inner_val = the other k-1 folds, split by
    `inner_val_frac`. Smoke trims all three to keep plumbing checks fast."""
    folds = folds_data["folds"]
    sigs = convert.class_signatures()
    test = list(folds[fold_idx])
    train_pool = [s for i, f in enumerate(folds) if i != fold_idx for s in f]
    train, inner_val = _carve_inner_val(train_pool, sigs, inner_val_frac, seed)
    if smoke:
        train = train[:S.SMOKE_SUBSET]
        inner_val = inner_val[:max(2, S.SMOKE_SUBSET // 4)]
        test = test[:max(2, S.SMOKE_SUBSET // 4)]
    return {"train": sorted(train), "inner_val": sorted(inner_val), "test": sorted(test)}


def materialise_fold(part: dict, tree: Path) -> Path:
    """Build a YOLO tree {images,labels}/{train,val} + data.yaml for one fold.

    train = part['train'], val = part['inner_val'] (inner-val for best.pt only).
    Symlinks to the read-only originals + shared converted labels. Returns the
    data.yaml path. Clears stale links first so the set is deterministic."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = tree / sub
        if d.exists():
            shutil.rmtree(d)

    for subset, ids in (("train", part["train"]), ("val", part["inner_val"])):
        for stem in ids:
            splits._link(splits.tb_image_path(stem), tree / "images" / subset / f"{stem}.png")
            splits._link(splits.label_path(stem), tree / "labels" / subset / f"{stem}.txt")

    yaml_path = tree / "data.yaml"
    yaml_path.write_text(
        f"# generated by yolo_common/kfold.py — rebuildable, do not hand-edit\n"
        f"path: {tree.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {S.NUM_CLASSES}\n"
        f"names: {S.CLASS_NAMES}\n"
    )
    return yaml_path


def fold_eval_inputs(part: dict) -> tuple[dict, list[tuple[Path, Path, str]]]:
    """(split_like, items) for evaluating on the rotating TEST fold.

    split_like carries test_positive_ids so metrics.detection_ap works unchanged;
    items is the positives-only blackbox list (no negatives → screening FA = 0)."""
    test = part["test"]
    split_like = {"test_positive_ids": test}
    items = [(splits.tb_image_path(s), splits.label_path(s), "tb") for s in test]
    return split_like, items


if __name__ == "__main__":
    import os
    k = int(os.environ.get("KFOLD_K", "5"))
    data = build_or_load_folds(k, S.SEED)
    print(json.dumps({"seed": data["seed"], "k": data["k"],
                      "n_positives": data["n_positives"],
                      "fold_sizes": data["fold_sizes"],
                      "fold_signatures": data["fold_signatures"]}, indent=2))
