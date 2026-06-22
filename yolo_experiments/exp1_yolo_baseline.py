"""
exp1_yolo_baseline.py — Unoptimised YOLOv8n TB detection FLOOR (diagnostic).

PURPOSE (diagnostic, not performance)
    A naive, COCO-pretrained YOLOv8n detector trained on TB positives only, with
    EVERY augmentation off, AMP off, no tuning, no CV. We deliberately do NOT
    help it. The point is to see exactly where a naive baseline fails — small-
    lesion misses, Active/Obsolete confusion, and false alarms on abnormal-but-
    not-TB (sick) lungs — so later experiments know precisely what to fix.

ABLATION (one file = one ablation; the only variable is input resolution)
    --imgsz 512   → exp1a_512
    --imgsz 1024  → exp1b_1024
    Both evaluate the SAME sealed 360-image blackbox; compare the two directly.

RUN
    python yolo_experiments/exp1_yolo_baseline.py --imgsz 512      # → yolov8n_512
    python yolo_experiments/exp1_yolo_baseline.py --imgsz 1024     # → yolov8n_1024
    TB_SMOKE=1 python yolo_experiments/exp1_yolo_baseline.py --imgsz 512   # plumbing
    # bigger model / shorter run / partial freeze, any combination:
    python yolo_experiments/exp1_yolo_baseline.py --imgsz 512 --model yolov8s.pt
    python yolo_experiments/exp1_yolo_baseline.py --imgsz 512 --epochs 40 --freeze 10

KNOBS — every one is an env var (settings.py) AND a CLI flag; CLI wins.
    --model (MODEL) --epochs (EPOCHS) --batch (BATCH) --device (DEVICE)
    --aug-level (AUG_LEVEL) --freeze (FREEZE) --patience (PATIENCE) --lr0 (LR0)
    --workers (WORKERS) --name (output run dir).  Also: SEED, CONF_THRESHOLDS,
    TB_DATA_ROOT / TB_GEN_ROOT / TB_RESULTS_ROOT (relocate data/in/out).
    TB_SMOKE=1 → 2 epochs on a tiny subset, full pipeline, no real train.
    Default run name auto-derives: <model>_<imgsz>[_fz<N>][_<aug>].

READING THE RESULTS  (yolo_experiments/results/<run-name>/)
    metrics.json  machine-readable, stable schema (renders uniformly on the web page)
    summary.txt   the human table
    <ultralytics run dir>  best.pt, PR/F1 curves, confusion matrix
    config_snapshot.json   exact knobs for this run
    Key signals: small-lesion recall (by_size.small), Active-vs-Obsolete gap
    (detection per class), and the healthy-vs-sick false-alarm GAP (screening).
    A positives-only model is EXPECTED to over-fire on negatives — high false
    alarm is the diagnostic, not a bug.

HALT-ON-CONTRADICTION
    Step 0 inventories the data and verifies it against data_research_report.md
    (folder counts, 799 positives, box hist 972 Active / 239 Obsolete / 1211
    total, class map 0=Active / 1=Obsolete). On any mismatch it STOPS — it does
    not guess or silently 'fix'.
"""

from __future__ import annotations

import argparse
import json
import sys
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
        description="exp1 YOLOv8 TB detection floor. Every knob is also an env "
                    "var (see settings.py); CLI flags take precedence.")
    ap.add_argument("--imgsz", type=int, required=True, help="input resolution")
    # All optional; None → fall back to the (env-overridable) settings default.
    ap.add_argument("--model", default=None, help="weights, e.g. yolov8n.pt / yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help='"0" GPU / "cpu"')
    ap.add_argument("--aug-level", default=None, help="off / light / default (aug.py)")
    ap.add_argument("--freeze", type=int, default=None, help="leading layers to freeze")
    ap.add_argument("--patience", type=int, default=None, help="early-stop patience")
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None,
                    help="output run name (default: auto from model/imgsz/freeze/aug)")
    args = ap.parse_args()

    # Resolve effective config (CLI > env/settings) for naming + the snapshot.
    model = args.model or S.MODEL
    aug_level = args.aug_level or S.AUG_LEVEL
    freeze = args.freeze if args.freeze is not None else S.FREEZE

    if args.name:
        tag = args.name
    else:
        tag = f"{Path(model).stem}_{args.imgsz}"
        if freeze:
            tag += f"_fz{freeze}"
        if aug_level != "off":
            tag += f"_{aug_level}"
    print(f"\n████ {tag}  (model={model}, imgsz={args.imgsz}, aug={aug_level}, "
          f"freeze={freeze}, smoke={S.SMOKE}) ████\n")

    conv_rep = inventory_or_halt()

    # Step 1 — frozen split (build once) + materialised train/val tree.
    split = splits.build_or_load()
    print(f"══ Step 1 — split (seed={split['seed']}) ══")
    print(f"  {json.dumps(split['counts'])}")
    data_yaml = splits.materialise(smoke=S.SMOKE)
    print(f"  data.yaml → {data_yaml}\n")

    # Step 2 — train (positives only, aug OFF, amp OFF, deterministic).
    out_dir = S.RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"══ Step 2 — train {model} @ imgsz={args.imgsz} ══")
    tr = train_eval.train(data_yaml, args.imgsz, project=out_dir, name="ultralytics",
                          model=args.model, epochs=args.epochs, batch=args.batch,
                          device=args.device, aug_level=args.aug_level,
                          freeze=args.freeze, patience=args.patience,
                          lr0=args.lr0, workers=args.workers)
    print(f"  best.pt → {tr['best_pt']}  ({tr['train_sec']}s)\n")

    # Step 3 — diagnostic eval on the sealed 360.
    print("══ Step 3 — evaluate on the sealed 360 ══")
    import time
    t0 = time.time()
    model = train_eval.load_best(tr["best_pt"])
    items = splits.blackbox(split)
    metric_block = metrics.evaluate(model, args.imgsz, split, items)
    eval_sec = round(time.time() - t0, 1)

    # Step 4 — assemble metrics.json (locked schema) + summary.txt + snapshot.
    report = {
        "experiment": tag,
        "config": {**tr["resolved"], "imgsz": args.imgsz, "seed": S.SEED,
                   "augmentation": tr["resolved"]["aug_level"] != "off", "amp": S.AMP,
                   "conf_thresholds": S.CONF_THRESHOLDS},
        "dataset": {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                    "train_positives": split["counts"]["train_positives"],
                    "val_positives": split["counts"]["val_positives"],
                    "blackbox": {"tb": split["counts"]["test_positives"],
                                 "sick_non_tb": split["counts"]["blackbox_sick"],
                                 "healthy": split["counts"]["blackbox_healthy"]},
                    "box_counts": {"active": conv_rep["box_counts"]["ActiveTuberculosis"],
                                   "obsolete": conv_rep["box_counts"]["ObsoletePulmonaryTuberculosis"]},
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
