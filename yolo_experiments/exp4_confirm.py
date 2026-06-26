"""
exp4_confirm.py — CONFIRM the exp3 finalist at full resolution, multi-seed.

PURPOSE
    exp3 screens augmentation levels single-seed at 512@16 to find the direction.
    exp4 confirms the chosen finalist at the quality ceiling this GPU allows
    (1024 @ batch 8) and across SEEDS, because exp2 taught us single-seed
    run-to-run noise can be as large as the effect. Two arms:

      --arm posonly   positives-only detector — stage 2 of the classifier→
                      detector pipeline (kept sensitive; this is the real model).
      --arm neg       + negatives in training — the SINGLE-STAGE YOLO baseline
                      (one detector forced to do detection AND specificity). This
                      is the strawman the two-stage system must beat; it is part
                      of the ablation, NOT a detector we expect to be better.

    Run BOTH arms across 3 seeds, then compare the seed-averaged Active mAP50.

MULTI-SEED (frozen split stays frozen; only the training seed changes)
    SEED is an env knob read at import. splits.json is loaded, never rebuilt, so
    the split is identical across seeds — only weight init, data order, and (for
    --arm neg) the negative draw vary. Run one seed per invocation:

        for s in 0 1 2; do
          SEED=$s python yolo_experiments/exp4_confirm.py --arm posonly --aug-level <finalist>
        done
        for s in 0 1 2; do
          SEED=$s python yolo_experiments/exp4_confirm.py --arm neg --aug-level <finalist>
        done

    Run seeds SEQUENTIALLY — arms share the generated tree dir.
    Default run name: exp4_<arm>_<aug>_<imgsz>_s<seed>  (override with --name).
    TB_SMOKE=1 ... --arm posonly --aug-level geo   # plumbing

NEGATIVES (--arm neg only; leakage-guarded — the 240 reserved ids are excluded)
    NEG_PER_POS / --neg-per-pos      negatives per train-positive (default 1.0)
    NEG_SICK_FRAC / --neg-sick-frac  sick fraction of them (default 0.5)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → yolo_common
from yolo_common import aug as AUG, metrics, settings as S, splits, train_eval
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
        description="exp4: confirm the exp3 finalist at 1024@8, multi-seed. "
                    "--arm posonly (the two-stage detector) or neg (single-stage "
                    "YOLO baseline). SEED env selects the seed (one per run).")
    ap.add_argument("--arm", required=True, choices=("posonly", "neg"))
    ap.add_argument("--aug-level", required=True, choices=sorted(AUG.AUG_LEVELS),
                    help="the exp3 finalist to confirm")
    ap.add_argument("--imgsz", type=int, default=1024, help="confirm default 1024")
    ap.add_argument("--batch", type=int, default=8, help="1024 caps at 8 on 6GB")
    ap.add_argument("--epochs", type=int, default=200)
    # negatives (--arm neg)
    ap.add_argument("--neg-per-pos", type=float, default=None)
    ap.add_argument("--neg-sick-frac", type=float, default=None)
    # exp1-style knobs (None → settings/env default)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    model = args.model or S.MODEL
    seed = S.SEED
    tag = args.name or f"exp4_{args.arm}_{args.aug_level}_{args.imgsz}_s{seed}"
    print(f"\n████ {tag}  (arm={args.arm}, model={model}, imgsz={args.imgsz}, "
          f"batch={args.batch}, epochs={args.epochs}, aug={args.aug_level}, "
          f"seed={seed}, smoke={S.SMOKE}) ████")
    print(f"  aug: {AUG.describe(args.aug_level)}\n")

    inv = inventory_or_halt()
    split = splits.build_or_load()
    print(f"══ Step 1 — split (seed={split['seed']}) + tree (arm={args.arm}) ══")
    print(f"  {json.dumps(split['counts'])}")

    neg_block = None
    if args.arm == "neg":
        neg_per_pos = args.neg_per_pos if args.neg_per_pos is not None else S.NEG_PER_POS
        neg_sick_frac = args.neg_sick_frac if args.neg_sick_frac is not None else S.NEG_SICK_FRAC
        n_neg_total = round(neg_per_pos * len(split["train_ids"]))
        # Fixed-seed, leakage-safe (reserved 240 excluded); varies per SEED.
        negs = splits.select_train_negatives(n_neg_total, sick_frac=neg_sick_frac,
                                             seed=seed, split=split)
        neg_stems = negs["sick"] + negs["healthy"]
        reserved = set(split["blackbox_negative_ids"]["sick"]) \
            | set(split["blackbox_negative_ids"]["healthy"])
        leaked = reserved & set(neg_stems)
        if leaked:
            raise SystemExit(f"LEAKAGE: {len(leaked)} reserved negatives selected — "
                             f"aborting. {sorted(leaked)[:5]}...")
        # Per-seed tree so independent seeds never clash on the same dir.
        tree = S.GEN_ROOT / f"det_neg_s{seed}"
        data_yaml = splits.materialise(tree=tree, train_negative_stems=neg_stems,
                                       smoke=S.SMOKE)
        neg_block = {"sick": len(negs["sick"]), "healthy": len(negs["healthy"]),
                     "total": len(neg_stems), "neg_per_pos": neg_per_pos,
                     "neg_sick_frac": neg_sick_frac}
        print(f"  + {len(neg_stems)} train negatives (sick {len(negs['sick'])}, "
              f"healthy {len(negs['healthy'])}) — reserved 240 excluded ✓")
    else:
        data_yaml = splits.materialise(smoke=S.SMOKE)     # positives only
        print("  positives-only (no negatives)")
    print(f"  data.yaml → {data_yaml}\n")

    # Step 2 — train.
    out_dir = S.RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"══ Step 2 — train {model} @ imgsz={args.imgsz}, aug={args.aug_level} ══")
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

    # Step 4 — assemble metrics.json (exp1 schema + arm/aug/seed) + summary.
    report = {
        "experiment": tag,
        "config": {**tr["resolved"], "imgsz": args.imgsz, "seed": seed, "arm": args.arm,
                   "augmentation": tr["resolved"]["aug_level"] != "off", "amp": S.AMP,
                   "aug_kwargs": AUG.AUG_LEVELS[args.aug_level],
                   "negatives_in_training": args.arm == "neg",
                   "conf_thresholds": S.CONF_THRESHOLDS},
        "dataset": {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                    "train_positives": split["counts"]["train_positives"],
                    "val_positives": split["counts"]["val_positives"],
                    "train_negatives": neg_block,
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
                      "results_csv": str(tr["run_dir"] / "results.csv"),
                      "confusion_matrix_val": str(tr["run_dir"] / "confusion_matrix.png"),
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


if __name__ == "__main__":
    main()
