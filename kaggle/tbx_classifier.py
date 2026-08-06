"""
tbx_classifier.py — the IMAGE-LEVEL classifier stage of the locked pipeline.

    classifier -> detector          (stage 3 was dropped 2026-08-05: only 5/121
                                     sealed test images carry both Active and
                                     Obsolete boxes, so one label per image costs
                                     0 Active boxes — see RESULTS.md)

WHAT IT IS
    ResNet-50, 4-way: healthy / sick_non_tb / active / obsolete.
    One model doing both jobs the pipeline needs:
      * the stage-1 GATE      — is this a TB image at all? (healthy+sick vs rest)
      * the Active/Obsolete   — the call that moved OFF the detector when the
        LABEL                   detector went --single-class.

WHY ResNet-50 AND NOT ConvNeXt/Mamba
    Forced, not chosen. The BYOL artifact is byol_resnet50_coco_n112120_ep18.pt
    (arch: resnet50); those weights do not load into anything else. Run
    INIT=imagenet for the honest control — "did the SSL chain beat a boring
    ImageNet baseline?" is the first question a reader will ask.

THE INIT SHORTCUT (why there is no VinDr training stage here)
    COCO -> BYOL(NIH 112k) -> VinDr is ALREADY TRAINED — it is the detector
    checkpoint weights/pretrain/vindr_byolinit_matched_s0.pt. Its 318
    `backbone.body.*` tensors are a torchvision resnet50 trunk with the prefix
    stripped. So this script fine-tunes ONE stage instead of three.
      INIT=byolvindr  weights/pretrain/vindr_byolinit_matched_s0.pt   (COCO->BYOL->VinDr)
      INIT=cocovindr  weights/pretrain/vindr_cocoinit_matched_s0.pt   (COCO->VinDr)
      INIT=byol       weights/pretrain/byol_resnet50_coco_n112120_ep18.pt
      INIT=imagenet   torchvision ImageNet-1k                          (the control)

⚠⚠ THE LEAKAGE RULE — the invariant a brand-new arm is most likely to break.
    blackbox_negative_ids (120 sick + 120 healthy) are RESERVED for the sealed
    test set. This classifier trains on ~7,400 negatives and would happily
    swallow them. `_assert_no_leak()` HALTS if even one appears in train/val.
    Do not "fix" that by removing the check.

⚠ MIXED IMAGES: 30 of 799 positives carry both Active and Obsolete boxes. A
    single-label classifier cannot express that, so they are labelled ACTIVE
    (clinically the actionable one, and Active dominates them 6:5 in the sealed
    set). Recorded as `mixed_as_active` in the report — it is a modelling
    choice, not a data fact.

⚠ Metric: report AUROC (threshold-free). Fixed-confidence accuracy on this
    project has produced a wrong read four times — see the memory
    research-objective-method, Trap 1.

USAGE (env knobs, house style — every one also a CLI flag)
    TB_SMOKE=1 python kaggle/tbx_classifier.py --name smoke     # CPU plumbing
    INIT=byolvindr EPOCHS=30 python kaggle/tbx_classifier.py --name cls_byolvindr
    INIT=imagenet  EPOCHS=30 python kaggle/tbx_classifier.py --name cls_imagenet_control
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── SELF-CONTAINED BY DESIGN — no yolo_common import ──────────────────────────
# Every other script in kaggle/ stands alone (vindr_retinanet.py only inserts its
# own dir), because a Kaggle cell is usually `%%writefile` + run, with no repo
# attached. So this file reads the two things it needs — the FROZEN splits.json
# and the COCO annotations — directly, and takes its paths from env:
#
#   TB_SPLITS_JSON  frozen split (REQUIRED — see the halt below)
#   TB_DATA_ROOT    TBX11K root: <root>/imgs/{tb,health,sick}, <root>/annotations/…
#   TB_RESULTS_ROOT where runs are written
#
# ⚠ The duplication of class_signatures() below is deliberate and is the one
#   place the repo's no-copy-paste rule is knowingly broken. Keep it a faithful
#   copy of yolo_common/convert.py::class_signatures — same {0,1} → active /
#   obsolete / both mapping, driven by the same COCO category ids.
DATA_ROOT = Path(os.environ.get("TB_DATA_ROOT", "data/TBX11K"))
SPLITS_JSON = Path(os.environ.get(
    "TB_SPLITS_JSON", os.environ.get("TB_GEN_ROOT", "yolo_datasets") + "/splits.json"))
RESULTS_ROOT = Path(os.environ.get("TB_RESULTS_ROOT", "yolo_experiments/results"))
ANN_JSON = DATA_ROOT / "annotations" / "json" / "TBX11K_trainval_only_tb.json"
SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("TB_SMOKE") == "1"
COCO_CAT_TO_YOLO = {1: 0, 2: 1}          # 1=ActiveTuberculosis, 2=Obsolete…

CLASSES = ["healthy", "sick_non_tb", "active", "obsolete"]


def load_split() -> dict:
    """The FROZEN split — loaded, never rebuilt.

    ⚠⚠ Rebuilding does NOT reproduce it. yolo_common/splits.py seeds its shuffle
    with hash(sig), and Python randomises string hashing per process, so a
    rebuild deals a DIFFERENT sealed test set — TB images sealed on the dev box
    would become training data here. There is no build path in this file at all;
    if the file is absent we halt."""
    if not SPLITS_JSON.exists():
        raise SystemExit(
            f"no splits.json at {SPLITS_JSON}\n"
            f"This script never builds one — a rebuild does not reproduce the frozen "
            f"split (hash() randomisation), so the run would use a different sealed "
            f"test set than every recorded result.\n"
            f"Upload the frozen splits.json and set TB_SPLITS_JSON to it.")
    return json.loads(SPLITS_JSON.read_text())


def class_signatures() -> dict[str, str]:
    """stem → {'active','obsolete','both'}, straight from the COCO JSON.

    ⚠ The COCO JSON is the box-coord/label source of truth, NOT the VOC XML
    (the XML is at original clinical resolution ~2840px). Faithful copy of
    yolo_common/convert.py::class_signatures."""
    if not ANN_JSON.exists():
        raise SystemExit(f"COCO annotations not found at {ANN_JSON} — check TB_DATA_ROOT")
    coco = json.loads(ANN_JSON.read_text())
    anns = {}
    for a in coco["annotations"]:
        if a["category_id"] in COCO_CAT_TO_YOLO:
            anns.setdefault(a["image_id"], set()).add(COCO_CAT_TO_YOLO[a["category_id"]])
    sig = {}
    for im in coco["images"]:
        ys = anns.get(im["id"])
        if not ys:
            continue          # 800 tb images, 799 carry boxes — the 1 is not a positive
        sig[Path(im["file_name"]).stem] = (
            "both" if ys == {0, 1} else ("active" if ys == {0} else "obsolete"))
    return sig


# Weights live somewhere else on Kaggle (/kaggle/input/tb-weights), so the
# directory is a knob. `--init-weights /abs/path.pt` overrides outright.
WEIGHTS_ROOT = Path(os.environ.get("TB_WEIGHTS_ROOT", "weights/pretrain"))
INITS = {
    "byolvindr": WEIGHTS_ROOT / "vindr_byolinit_matched_s0.pt",
    "cocovindr": WEIGHTS_ROOT / "vindr_cocoinit_matched_s0.pt",
    "byol": WEIGHTS_ROOT / "byol_resnet50_coco_n112120_ep18.pt",
    "imagenet": None,
}


# ── data ──────────────────────────────────────────────────────────────────────
def _tb_label(stem: str, sigs: dict) -> int:
    """active=2 / obsolete=3. Mixed -> active (see the module docstring)."""
    sig = sigs.get(stem, "active")
    return CLASSES.index("obsolete") if sig == "obsolete" else CLASSES.index("active")


def build_sets(smoke: bool = False) -> dict:
    """train / val / test image lists as (path, label).

    Positives follow the FROZEN splits.json exactly, so this classifier and the
    detector never disagree about which TB images are sealed. Negatives are
    drawn from the pools MINUS the reserved blackbox ids, then split by the same
    train/val proportion the positives use."""
    sp = load_split()
    sigs = class_signatures()
    reserved = {g: set(sp["blackbox_negative_ids"][g]) for g in ("sick", "healthy")}

    def negs(group: str) -> list[str]:
        root = DATA_ROOT / "imgs" / ("health" if group == "healthy" else "sick")
        return sorted(p.stem for p in root.glob("*.png") if p.stem not in reserved[group])

    out = {"train": [], "val": [], "test": []}
    img = lambda g, s: DATA_ROOT / "imgs" / g / f"{s}.png"

    for key, dest in (("train_ids", "train"), ("val_ids", "val"),
                      ("test_positive_ids", "test")):
        for stem in sp[key]:
            out[dest].append((str(img("tb", stem)), _tb_label(stem, sigs)))

    # val gets the same share of negatives as positives do, so the val class mix
    # matches the train mix and best.pt is not selected on a skewed slice.
    frac = len(sp["val_ids"]) / (len(sp["train_ids"]) + len(sp["val_ids"]))
    for group, folder, lab in (("healthy", "health", 0), ("sick", "sick", 1)):
        pool = negs(group)
        n_val = int(round(len(pool) * frac))
        for s in pool[:n_val]:
            out["val"].append((str(img(folder, s)), lab))
        for s in pool[n_val:]:
            out["train"].append((str(img(folder, s)), lab))
        for s in sorted(reserved[group]):
            out["test"].append((str(img(folder, s)), lab))

    if smoke:
        out = {k: v[::max(1, len(v) // 24)][:24] for k, v in out.items()}
    _assert_no_leak(out, reserved)
    return out


def _assert_no_leak(sets: dict, reserved: dict) -> None:
    """HALT if a reserved blackbox negative reached train or val.

    ⚠ This is the leakage rule from CLAUDE.md and splits.json, enforced in code
    because it is invisible at training time — a leaked run trains fine, scores
    better, and the number is worthless. Halt-on-contradiction, house style."""
    bad = {g for g in reserved for path, _ in sets["train"] + sets["val"]
           if Path(path).stem in reserved[g]}
    if bad:
        raise SystemExit(
            f"LEAK: reserved blackbox negatives ({', '.join(sorted(bad))}) appeared in "
            f"train/val. These 240 ids are sealed — see splits.json 'leakage_rule'. "
            f"Refusing to train; every downstream number would be contaminated.")


class CxrSet(Dataset):
    """512px grayscale-as-RGB. Augmentation is flip + small affine + jitter — the
    same conservative level the detector's `geo` uses, since the finding all
    project has been that recipe knobs do not move this data."""

    def __init__(self, items, imgsz: int, train: bool):
        from torchvision import transforms as T
        self.items = items
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tf = T.Compose(
            ([T.RandomHorizontalFlip(),
              T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.92, 1.08)),
              T.ColorJitter(brightness=0.2, contrast=0.2)] if train else [])
            + [T.Resize((imgsz, imgsz)), T.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


# ── model ─────────────────────────────────────────────────────────────────────
def build_model(init: str, device, override: str | None = None) -> tuple[nn.Module, dict]:
    """ResNet-50 + a 4-way head, trunk initialised per `init`.

    Detector checkpoints carry the trunk as `backbone.body.*` (318 tensors);
    stripping that prefix yields a torchvision resnet50 state dict. `fc` is
    always missing — that is the head we are replacing, not a bug."""
    from torchvision import models
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if init == "imagenet" else None
    model = models.resnet50(weights=weights)
    src = Path(override) if override else INITS[init]
    meta = {"init": init, "source": str(src) if src else "torchvision ImageNet1K_V2"}

    if init != "imagenet":
        path = Path(src)
        if not path.exists():
            raise SystemExit(f"init weights not found: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        body = {k[len("backbone.body."):]: v for k, v in sd.items()
                if k.startswith("backbone.body.")}
        if not body:   # a raw BYOL trunk is already bare resnet keys
            body = {k: v for k, v in sd.items() if not k.startswith(("fc.", "head."))}
        missing, unexpected = model.load_state_dict(body, strict=False)
        loaded = len(body) - len(unexpected)
        meta |= {"tensors_loaded": loaded, "unexpected": len(unexpected),
                 "missing": [m for m in missing if not m.startswith("fc.")],
                 "vindr_meta": ck.get("vindr_meta") if isinstance(ck, dict) else None}
        # A silent fallback to random/ImageNet weights would look like a real run
        # and score like a bad one. Halt instead.
        if loaded < 300:
            raise SystemExit(f"only {loaded} trunk tensors loaded from {path} — expected "
                             f"~318. Refusing to train on a half-initialised trunk.")
        print(f"[init] {init}: {loaded} trunk tensors loaded, {len(unexpected)} unexpected, "
              f"{len(meta['missing'])} missing (excl. fc)")
    else:
        print("[init] imagenet: torchvision ResNet50_Weights.IMAGENET1K_V2 (CONTROL arm)")

    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(device), meta


# ── metrics ───────────────────────────────────────────────────────────────────
def auroc(scores: np.ndarray, pos: np.ndarray) -> float:
    """Threshold-free — rank-based, no sklearn dependency on Kaggle."""
    if pos.sum() in (0, len(pos)):
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    tb_score = P[:, 2] + P[:, 3]           # the stage-1 gate score
    is_tb = Y >= 2
    m = {
        "auroc_tb_vs_all": round(auroc(tb_score, is_tb), 4),
        "auroc_tb_vs_healthy": None, "auroc_tb_vs_sick": None,
        "acc4": round(float((P.argmax(1) == Y).mean()), 4),
        "n": int(len(Y)), "per_class_n": {CLASSES[c]: int((Y == c).sum()) for c in range(4)},
    }
    for name, neg in (("healthy", 0), ("sick", 1)):
        sel = is_tb | (Y == neg)
        m[f"auroc_tb_vs_{name}"] = round(auroc(tb_score[sel], is_tb[sel]), 4)
    # ⚠ The TB-vs-SICK number is the one that matters and the one that has always
    # been weakest: telling TB from other pathology is the real task; telling it
    # from healthy lungs is easy. Do not quote the vs_all number alone.
    if is_tb.sum():
        sub = P[is_tb][:, 2:]
        m["active_vs_obsolete_acc"] = round(float(
            (sub.argmax(1) + 2 == Y[is_tb]).mean()), 4)
    return m, P, Y


def gate_table(P: np.ndarray, Y: np.ndarray, targets=(0.95, 0.98, 1.00)) -> list[dict]:
    """What the stage-1 gate COSTS at a chosen sensitivity.

    ⚠ Trap 2: the gate can only ever LOSE recall for the pipeline (it filters
    images before the detector sees them). So the question is never "how good is
    it" but "what specificity do we buy for the TB images we drop"."""
    tb_score, is_tb = P[:, 2] + P[:, 3], Y >= 2
    rows = []
    for t in targets:
        pos = np.sort(tb_score[is_tb])
        thr = pos[max(0, int(np.floor((1 - t) * len(pos))) - 1)] if t < 1.0 else pos[0]
        keep = tb_score >= thr
        rows.append({
            "target_sensitivity": t, "threshold": round(float(thr), 4),
            "tb_kept": f"{int(keep[is_tb].sum())}/{int(is_tb.sum())}",
            "tb_dropped": int((~keep[is_tb]).sum()),
            "healthy_rejected": round(float((~keep[Y == 0]).mean()), 4),
            "sick_rejected": round(float((~keep[Y == 1]).mean()), 4),
        })
    return rows


# ── train ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="TBX11K image-level classifier (stage 1).")
    ap.add_argument("--init", choices=list(INITS), default=os.environ.get("INIT", "byolvindr"))
    ap.add_argument("--init-weights", default=os.environ.get("INIT_WEIGHTS") or None,
                    help="explicit path to the init checkpoint, overriding --init's "
                         "default location (TB_WEIGHTS_ROOT). Use on Kaggle: "
                         "--init-weights /kaggle/input/tb-weights/vindr_byolinit_matched_s0.pt")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "30")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", "32")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "3e-4")))
    ap.add_argument("--imgsz", type=int, default=int(os.environ.get("IMGSZ", "512")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    ap.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", "8")))
    ap.add_argument("--device", default=os.environ.get("DEVICE") or None)
    ap.add_argument("--name", default=os.environ.get("RUN_NAME") or None)
    args = ap.parse_args()

    smoke = SMOKE
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = args.name or f"cls_{args.init}_s{SEED}"
    epochs = 2 if smoke else args.epochs
    print(f"\n████ {tag}  (resnet50, init={args.init}, imgsz={args.imgsz}, "
          f"batch={args.batch}, epochs={epochs}, device={device}, smoke={smoke}) ████\n")

    sets = build_sets(smoke)
    for k, v in sets.items():
        print(f"  {k:<5} n={len(v):<5} {dict(Counter(CLASSES[l] for _, l in v))}")
    print("  [leak-check] no reserved blackbox negative in train/val ✓\n")

    tr_ds = CxrSet(sets["train"], args.imgsz, train=True)
    # 4-way imbalance is ~3.7k : 3.7k : 500 : 120, so a plain shuffle would show
    # the head ~1 obsolete image per 60. Sample inversely to class frequency.
    freq = Counter(l for _, l in sets["train"])
    w = [1.0 / freq[l] for _, l in sets["train"]]
    loaders = {
        # drop_last only when there is more than one batch to drop — otherwise a
        # dataset smaller than `batch` yields ZERO batches and the run "trains"
        # to loss 0.0000 without ever stepping the optimizer.
        "train": DataLoader(tr_ds, batch_size=args.batch, num_workers=args.workers,
                            sampler=WeightedRandomSampler(w, len(w), replacement=True),
                            pin_memory=device.type == "cuda",
                            drop_last=len(tr_ds) > args.batch),
        **{k: DataLoader(CxrSet(sets[k], args.imgsz, train=False), batch_size=args.batch,
                         shuffle=False, num_workers=args.workers)
           for k in ("val", "test")},
    }

    model, init_meta = build_model(args.init, device, args.init_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_ep, since, hist = -1.0, 0, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(y)
            n += len(y)
        sched.step()
        vm, _, _ = evaluate(model, loaders["val"], device)
        # Select on AUROC, not accuracy: accuracy at a fixed argmax is a
        # fixed-threshold number and this project has been burned by those.
        score = vm["auroc_tb_vs_sick"]
        hist.append({"epoch": ep, "train_loss": round(tot / max(1, n), 4), **vm})
        print(f"  ep{ep:>3}  loss {tot / max(1, n):.4f}  val AUROC(tb|sick) {score:.4f}  "
              f"AUROC(tb|all) {vm['auroc_tb_vs_all']:.4f}  acc4 {vm['acc4']:.4f}")
        if score > best:
            best, best_ep, since = score, ep, 0
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "init": init_meta, "epoch": ep}, out_dir / "best.pt")
        else:
            since += 1
            if args.patience and since >= args.patience:
                print(f"  early stop at ep{ep} (no val gain for {args.patience})")
                break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state_dict"])
    tm, P, Y = evaluate(model, loaders["test"], device)
    gates = gate_table(P, Y)

    report = {
        "experiment": tag,
        "config": {"arch": "resnet50", "init": args.init, "init_meta": init_meta,
                   "imgsz": args.imgsz, "batch": args.batch, "lr": args.lr,
                   "epochs_cap": epochs, "epochs_ran": len(hist), "best_epoch": best_ep,
                   "seed": SEED, "classes": CLASSES, "mixed_as_active": True,
                   "stage": "1 (gate + active/obsolete label) of classifier->detector"},
        "dataset": {k: {"n": len(v), "by_class": dict(Counter(CLASSES[l] for _, l in v))}
                    for k, v in sets.items()},
        "training": {"best_val_auroc_tb_vs_sick": round(best, 4), "history": hist},
        "metrics": {"test": tm, "gate": gates},
        "timing": {"train_sec": round(time.time() - t0, 1)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\n== SEALED TEST (121 tb + 120 sick + 120 healthy) ==")
    print(f"  AUROC tb-vs-sick    {tm['auroc_tb_vs_sick']}   <- the real task")
    print(f"  AUROC tb-vs-healthy {tm['auroc_tb_vs_healthy']}")
    print(f"  AUROC tb-vs-all     {tm['auroc_tb_vs_all']}")
    print(f"  active-vs-obsolete  {tm.get('active_vs_obsolete_acc')} (on the 121 tb images)")
    print("\n== GATE COST (stage 1 can only LOSE recall — what do we buy?) ==")
    for g in gates:
        print(f"  sens {g['target_sensitivity']:.2f}: keep {g['tb_kept']} tb  "
              f"(drop {g['tb_dropped']})  reject healthy {g['healthy_rejected']:.3f}  "
              f"sick {g['sick_rejected']:.3f}")
    print(f"\n→ {out_dir}/metrics.json\n")


if __name__ == "__main__":
    main()
