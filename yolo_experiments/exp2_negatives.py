"""
exp2_negatives.py — exp1 floor + negatives in TRAINING (single-variable test).

PURPOSE
    exp1 trained on TB positives only and (by construction) over-fired on
    abnormal-but-non-TB (sick) lungs — sick false-alarm ran ~2-3× the healthy
    rate. exp2 changes exactly ONE thing: it adds background (no-TB) images to
    training. Everything else is held at the exp1 floor — yolov8n, imgsz 512,
    batch 16, augmentation OFF, full fine-tune (no freeze), AMP OFF,
    deterministic, same frozen split, same sealed 360 blackbox.

    Question: how much do training negatives cut the false-alarm rate (and the
    sick-vs-healthy gap), and at what cost to TB recall?

NEGATIVES (drawn fixed-seed; the 240 reserved blackbox negatives are EXCLUDED)
    NEG_PER_POS   negatives added per train-positive (default 1.0 → ~559)
    NEG_SICK_FRAC fraction of them that are sick vs healthy (default 0.5)
    Added to TRAIN only, as empty-label background images. val stays
    positives-only so best-weight selection is comparable to exp1; the
    false-alarm effect is measured on the sealed blackbox.

RUN (the floor config: 512 @ batch 16)
    python yolo_experiments/exp2_negatives.py --imgsz 512 --batch 16
    # sweep the ratio:
    NEG_PER_POS=3 python yolo_experiments/exp2_negatives.py --imgsz 512 --batch 16
    TB_SMOKE=1 python yolo_experiments/exp2_negatives.py --imgsz 512   # plumbing

    Default run name: exp2_neg<ratio>_<imgsz>  (override with --name).

READING
    Compare metrics.json against exp1's yolov8n_512 (batch 8) / 512@16 — same
    blackbox, same schema. Watch: screening sick_false_alarm and the
    healthy-vs-sick gap (should fall), vs tb_detect_rate / recall (cost).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → yolo_common
from yolo_common import metrics, settings as S, splits, train_eval
from yolo_common.inventory import inventory_or_halt


def environment() -> dict:
    info = {"timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": "n/a (not a git repo)"}
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="exp2: exp1 floor + negatives in training. Knobs mirror "
                    "exp1; plus NEG_PER_POS / NEG_SICK_FRAC (env) and the flags below.")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--neg-per-pos", type=float, default=None,
                    help="negatives per train-positive (default env NEG_PER_POS=1.0)")
    ap.add_argument("--neg-sick-frac", type=float, default=None,
                    help="fraction of negatives that are sick (default env NEG_SICK_FRAC=0.5)")
    # exp1-style knobs (None → settings/env default)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--aug-level", default=None)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    neg_per_pos = args.neg_per_pos if args.neg_per_pos is not None else S.NEG_PER_POS
    neg_sick_frac = args.neg_sick_frac if args.neg_sick_frac is not None else S.NEG_SICK_FRAC
    model = args.model or S.MODEL

    tag = args.name or f"exp2_neg{neg_per_pos:g}_{args.imgsz}"
    print(f"\n████ {tag}  (model={model}, imgsz={args.imgsz}, "
          f"neg_per_pos={neg_per_pos}, sick_frac={neg_sick_frac}, smoke={S.SMOKE}) ████\n")

    inv = inventory_or_halt()

    # Step 1 — split + negative selection (leakage-safe: reserved 240 excluded).
    split = splits.build_or_load()
    n_pos = len(split["train_ids"])
    n_neg_total = round(neg_per_pos * n_pos)
    negs = splits.select_train_negatives(n_neg_total, sick_frac=neg_sick_frac, split=split)
    neg_stems = negs["sick"] + negs["healthy"]

    # Defensive leakage guard — selected negatives must not touch the blackbox.
    reserved = set(split["blackbox_negative_ids"]["sick"]) \
        | set(split["blackbox_negative_ids"]["healthy"])
    leaked = reserved & set(neg_stems)
    if leaked:
        raise SystemExit(f"LEAKAGE: {len(leaked)} reserved negatives selected for "
                         f"training — aborting. {sorted(leaked)[:5]}...")

    print(f"══ Step 1 — split + negatives ══")
    print(f"  train positives: {n_pos}  | val positives: {len(split['val_ids'])}")
    print(f"  train negatives: {len(neg_stems)}  (sick {len(negs['sick'])}, "
          f"healthy {len(negs['healthy'])})  — reserved 240 excluded ✓")

    tree = S.GEN_ROOT / "det_neg"
    data_yaml = splits.materialise(tree=tree, train_negative_stems=neg_stems, smoke=S.SMOKE)
    print(f"  data.yaml → {data_yaml}\n")

    # Step 2 — train (floor config + negatives).
    out_dir = S.RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"══ Step 2 — train {model} @ imgsz={args.imgsz} (+{len(neg_stems)} negatives) ══")
    tr = train_eval.train(data_yaml, args.imgsz, project=out_dir, name="ultralytics",
                          model=args.model, epochs=args.epochs, batch=args.batch,
                          device=args.device, aug_level=args.aug_level,
                          freeze=args.freeze, patience=args.patience,
                          lr0=args.lr0, workers=args.workers)
    print(f"  best.pt → {tr['best_pt']}  ({tr['train_sec']}s)\n")

    # Step 3 — diagnostic eval on the same sealed 360.
    print("══ Step 3 — evaluate on the sealed 360 ══")
    t0 = time.time()
    yolo = train_eval.load_best(tr["best_pt"])
    items = splits.blackbox(split)
    metric_block = metrics.evaluate(yolo, args.imgsz, split, items)
    eval_sec = round(time.time() - t0, 1)

    # Step 4 — assemble metrics.json (exp1 schema + negatives block) + summary.
    report = {
        "experiment": tag,
        "config": {**tr["resolved"], "imgsz": args.imgsz, "seed": S.SEED,
                   "augmentation": tr["resolved"]["aug_level"] != "off", "amp": S.AMP,
                   "conf_thresholds": S.CONF_THRESHOLDS,
                   "negatives_in_training": True,
                   "neg_per_pos": neg_per_pos, "neg_sick_frac": neg_sick_frac},
        "dataset": {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                    "train_positives": n_pos, "val_positives": len(split["val_ids"]),
                    "train_negatives": {"sick": len(negs["sick"]),
                                        "healthy": len(negs["healthy"]),
                                        "total": len(neg_stems)},
                    "blackbox": {"tb": split["counts"]["test_positives"],
                                 "sick_non_tb": split["counts"]["blackbox_sick"],
                                 "healthy": split["counts"]["blackbox_healthy"]},
                    "box_counts": {"active": inv["box_counts"]["ActiveTuberculosis"],
                                   "obsolete": inv["box_counts"]["ObsoletePulmonaryTuberculosis"]},
                    "class_map": {"0": "ActiveTuberculosis", "1": "ObsoletePulmonaryTuberculosis"}},
        "environment": environment(),
        "metrics": metric_block,
        "timing": {"train_sec": tr["train_sec"], "eval_sec": eval_sec},
        "artifacts": {"weights": str(tr["best_pt"]),
                      "pr_curve": str(tr["run_dir"] / "PR_curve.png"),
                      "confusion_matrix": str(tr["run_dir"] / "confusion_matrix.png")},
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "summary.txt").write_text(metrics.summary_text(report))
    (out_dir / "config_snapshot.json").write_text(json.dumps(report["config"], indent=2))

    print(f"\n→ {out_dir}/metrics.json")
    print(f"→ {out_dir}/summary.txt\n")
    print((out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
