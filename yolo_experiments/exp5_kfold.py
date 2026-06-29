"""
exp5_kfold.py — Stratified k-fold CV of the FINAL detector config.

PURPOSE
    exp4 locked the detector config (mosaic @ 512 @ batch 16) on the single
    frozen split. exp5 asks the one question a single split can't answer: does
    that result survive WHICH images are the test set, or did we anchor on a
    lucky split? It rotates a stratified test fold over ALL ~799 positives so
    every positive is tested exactly once, and reports Active mAP50 mean±std
    across the k folds.

    This is a robustness MEASUREMENT, not a model we ship — the k fold models
    are thrown away. The frozen splits.json and its sealed blackbox are NOT
    touched (exp1-4/exp6 stay reproducible); exp5 builds its own partition in
    yolo_datasets/kfold/ (see yolo_common/kfold.py).

CONFIG (locked — the exp4 finalist; not a tuning run)
    positives-only · aug=mosaic · imgsz=512 · batch=16 · 200 epochs · single
    seed per fold. Inner-val (stratified ~15% of the k-1 training folds) selects
    best.pt; the rotating test fold is the held-out eval.

HOW TO RUN  (one fold per invocation — the user owns the GPU; folds share one
            partition built on the first call, so order doesn't matter)
    for f in 0 1 2 3 4; do
      python yolo_experiments/exp5_kfold.py --fold $f
    done
    python yolo_experiments/exp5_kfold.py --aggregate    # mean±std across folds

    TB_SMOKE=1 python yolo_experiments/exp5_kfold.py --fold 0   # plumbing only

KNOBS (env or flag; flag wins). SEED selects the partition seed AND train seed.
    --k (KFOLD_K=5) · --inner-val-frac (KFOLD_INNER_VAL_FRAC=0.15) ·
    --imgsz 512 · --batch 16 · --epochs 200 · --aug-level mosaic · plus the
    exp1-style --model/--device/--freeze/--patience/--lr0/--workers passthroughs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → yolo_common
from yolo_common import aug as AUG, kfold, metrics, settings as S, splits, train_eval
from yolo_common.inventory import inventory_or_halt


def environment() -> dict:
    info = {"timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = "?"
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["torch"], info["gpu"] = "?", "?"
    return info


def fold_tag(fold_idx: int, seed: int) -> str:
    return f"exp5_fold{fold_idx}_s{seed}"


def run_fold(args, fold_idx: int) -> None:
    seed = S.SEED
    k = args.k
    folds_data = kfold.build_or_load_folds(k, seed)
    if not (0 <= fold_idx < k):
        raise SystemExit(f"--fold must be in [0, {k}); got {fold_idx}")

    aug_level = args.aug_level
    tag = args.name or fold_tag(fold_idx, seed)
    print(f"\n████ {tag}  (fold {fold_idx}/{k - 1}, model={args.model or S.MODEL}, "
          f"imgsz={args.imgsz}, batch={args.batch}, epochs={args.epochs}, "
          f"aug={aug_level}, seed={seed}, smoke={S.SMOKE}) ████")
    print(f"  aug: {AUG.describe(aug_level)}")

    inv = inventory_or_halt()

    print(f"══ Step 1 — fold partition (seed={seed}, k={k}) ══")
    print(f"  fold sizes: {folds_data['fold_sizes']}  (Σ={folds_data['n_positives']})")
    part = kfold.fold_partition(folds_data, fold_idx, inner_val_frac=args.inner_val_frac,
                                seed=seed, smoke=S.SMOKE)
    print(f"  train={len(part['train'])}  inner_val={len(part['inner_val'])}  "
          f"test(fold)={len(part['test'])}  (positives-only, no negatives)")

    tree = kfold.KFOLD_ROOT / f"fold{fold_idx}"
    data_yaml = kfold.materialise_fold(part, tree)
    print(f"  data.yaml → {data_yaml}\n")

    # Step 2 — train the finalist config on this fold.
    out_dir = S.RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"══ Step 2 — train @ imgsz={args.imgsz}, aug={aug_level} ══")
    tr = train_eval.train(data_yaml, args.imgsz, project=out_dir, name="ultralytics",
                          model=args.model, epochs=args.epochs, batch=args.batch,
                          device=args.device, aug_level=aug_level,
                          freeze=args.freeze, patience=args.patience,
                          lr0=args.lr0, workers=args.workers)
    print(f"  best.pt → {tr['best_pt']}  ({tr['train_sec']}s)\n")

    # Step 3 — evaluate on the rotating TEST fold (fold-local eval tree).
    print("══ Step 3 — evaluate on the held-out test fold ══")
    t0 = time.time()
    yolo = train_eval.load_best(tr["best_pt"])
    split_like, items = kfold.fold_eval_inputs(part)
    eval_root = kfold.KFOLD_ROOT / f"fold{fold_idx}_eval"
    metric_block = metrics.evaluate(yolo, args.imgsz, split_like, items, eval_root=eval_root)
    eval_sec = round(time.time() - t0, 1)

    # Step 4 — metrics.json (exp4 schema, fold-flavoured) + summary.
    report = {
        "experiment": tag,
        "config": {**tr["resolved"], "imgsz": args.imgsz, "seed": seed,
                   "arm": "kfold-posonly", "fold": fold_idx, "k": k,
                   "inner_val_frac": args.inner_val_frac,
                   "augmentation": tr["resolved"]["aug_level"] != "off", "amp": S.AMP,
                   "aug_kwargs": AUG.AUG_LEVELS[aug_level],
                   "negatives_in_training": False,
                   "conf_thresholds": S.CONF_THRESHOLDS},
        "dataset": {"folds_file": str(kfold.FOLDS_JSON), "fold_seed": seed, "k": k,
                    "fold_sizes": folds_data["fold_sizes"],
                    "fold_signatures": folds_data["fold_signatures"],
                    "this_fold": {"train": len(part["train"]),
                                  "inner_val": len(part["inner_val"]),
                                  "test": len(part["test"])},
                    "box_counts": {"active": inv["box_counts"]["ActiveTuberculosis"],
                                   "obsolete": inv["box_counts"]["ObsoletePulmonaryTuberculosis"]},
                    "class_map": {"0": "ActiveTuberculosis", "1": "ObsoletePulmonaryTuberculosis"}},
        "environment": environment(),
        "metrics": metric_block,
        "timing": {"train_sec": tr["train_sec"], "eval_sec": eval_sec},
        "artifacts": {"weights": str(tr["best_pt"]),
                      "pr_curve": str(tr["run_dir"] / "PR_curve.png"),
                      "results_csv": str(tr["run_dir"] / "results.csv"),
                      "confusion_matrix_test": str(out_dir / "confusion_matrix_test.png")},
    }
    metrics.save_confusion_png(metric_block["confusion_matrix"],
                               out_dir / "confusion_matrix_test.png")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "summary.txt").write_text(metrics.summary_text(report))
    (out_dir / "config_snapshot.json").write_text(json.dumps(report["config"], indent=2))

    print(f"\n→ {out_dir}/metrics.json")
    print(f"→ {out_dir}/summary.txt\n")
    print((out_dir / "summary.txt").read_text())


def _mean_std(xs: list[float]) -> dict:
    a = np.array(xs, dtype=float)
    return {"mean": round(float(a.mean()), 4),
            # sample std (ddof=1) — the spread across folds, not of a population
            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "per_fold": [round(float(x), 4) for x in a]}


def aggregate(args) -> None:
    seed = S.SEED
    k = args.k
    print(f"══ exp5 aggregate — k={k}, seed={seed} ══")
    folds_meta, missing = [], []
    for i in range(k):
        # aggregate always uses the default per-fold tag (--name is single-fold only)
        p = S.RESULTS_ROOT / fold_tag(i, seed) / "metrics.json"
        if not p.exists():
            missing.append(i)
            continue
        folds_meta.append((i, json.loads(p.read_text())))
    if missing:
        raise SystemExit(f"missing fold results {missing} — run those folds first "
                         f"(looked under {S.RESULTS_ROOT}/exp5_fold<i>_s{seed}/).")

    def col(path):
        out = []
        for _, m in folds_meta:
            cur = m["metrics"]
            for key in path:
                cur = cur[key]
            out.append(cur)
        return out

    summary = {
        "experiment": f"exp5_kfold_s{seed}",
        "k": k, "seed": seed,
        "folds": [i for i, _ in folds_meta],
        "headline_active_mAP50": _mean_std(col(["detection", "active", "mAP50"])),
        "overall_mAP50": _mean_std(col(["detection", "overall", "mAP50"])),
        "obsolete_mAP50_noise": _mean_std(col(["detection", "obsolete", "mAP50"])),
        "active_recall": _mean_std(col(["detection", "active", "recall"])),
        "guardrail_medium_recall": _mean_std(col(["by_size", "medium", "recall"])),
        "guardrail_loc_iou_active": _mean_std(col(["localization", "active", "iou"])),
        "config": folds_meta[0][1]["config"],
        "environment": environment(),
    }
    out_dir = S.RESULTS_ROOT / f"exp5_kfold_s{seed}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    h = summary["headline_active_mAP50"]
    lines = [
        f"# exp5 k-fold CV (k={k}, seed={seed}) — config: mosaic @ {summary['config']['imgsz']} "
        f"@ batch {summary['config']['batch']}, {summary['config']['epochs']}ep, positives-only",
        f"# every positive tested exactly once; numbers are mean±std across {len(folds_meta)} folds",
        "",
        f"  HEADLINE  Active mAP50   {h['mean']:.3f} ± {h['std']:.3f}   "
        f"(folds {h['per_fold']}, range {h['min']:.3f}–{h['max']:.3f})",
    ]
    for label, key in (("overall mAP50", "overall_mAP50"),
                       ("Active recall", "active_recall"),
                       ("Obsolete mAP50 (noise)", "obsolete_mAP50_noise"),
                       ("medium recall (guard)", "guardrail_medium_recall"),
                       ("loc IoU active (guard)", "guardrail_loc_iou_active")):
        b = summary[key]
        lines.append(f"  {label:<24} {b['mean']:.3f} ± {b['std']:.3f}   "
                     f"{b['per_fold']}")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text)
    print(f"\n→ {out_dir}/summary.json\n→ {out_dir}/summary.txt\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="exp5: stratified k-fold CV of the final detector config "
                    "(mosaic@512@16). One fold per invocation, then --aggregate.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int, help="outer test fold index [0, k)")
    g.add_argument("--aggregate", action="store_true",
                   help="collect per-fold metrics → mean±std")
    ap.add_argument("--k", type=int, default=int(os.environ.get("KFOLD_K", "5")))
    ap.add_argument("--inner-val-frac", type=float,
                    default=float(os.environ.get("KFOLD_INNER_VAL_FRAC", "0.15")))
    ap.add_argument("--aug-level", default="mosaic", choices=sorted(AUG.AUG_LEVELS),
                    help="finalist; default mosaic (the exp4 winner)")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    # exp1-style passthroughs (None → settings/env default)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None, help="override run name (single fold only)")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args)
    else:
        run_fold(args, args.fold)


if __name__ == "__main__":
    main()
