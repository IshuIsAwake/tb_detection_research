"""
eval_weights.py — Evaluate an ALREADY-TRAINED best.pt on the sealed test.

Use case: training was offloaded to Kaggle/Colab (kaggle/tbx_train.py) at a size
the local card can't train (1024@16, yolov8s). Download the best.pt and run THIS
locally — inference on the sealed 360 fits in 6 GB even when training doesn't, so
the numbers are produced by the SAME yolo_common/metrics.py as exp1-6 and are
directly comparable. No training here; eval only.

    python yolo_experiments/eval_weights.py --weights /path/to/best.pt \
        --imgsz 1024 --name tbx_yolov8n_mosaic_1024_b16_s0

Writes results/<name>/metrics.json + summary.txt in the exp1-6 schema, so the
existing comparison tooling reads it with no changes. The frozen splits.json is
loaded (never rebuilt) — the blackbox is identical to every prior experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import metrics, settings as S, splits, train_eval


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eval a trained best.pt on the sealed test (for Kaggle/Colab-"
                    "trained weights). Eval only — no training.")
    ap.add_argument("--weights", required=True, help="path to the downloaded best.pt")
    ap.add_argument("--imgsz", type=int, required=True,
                    help="MUST match the training imgsz (e.g. 1024)")
    ap.add_argument("--name", required=True, help="results/<name>/ to write")
    ap.add_argument("--meta", default=None,
                    help="optional JSON string of training config to record "
                         "(model, batch, aug, seed, epochs) for provenance")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    print(f"\n████ eval {args.name}  (weights={weights.name}, imgsz={args.imgsz}) ████\n")
    split = splits.build_or_load()
    print(f"  frozen split seed={split['seed']} | blackbox "
          f"tb={split['counts']['test_positives']} "
          f"sick={split['counts']['blackbox_sick']} "
          f"healthy={split['counts']['blackbox_healthy']}")

    t0 = time.time()
    yolo = train_eval.load_best(weights)
    items = splits.blackbox(split)
    metric_block = metrics.evaluate(yolo, args.imgsz, split, items)
    eval_sec = round(time.time() - t0, 1)

    out_dir = S.RESULTS_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(args.meta) if args.meta else {}
    # summary_text expects config['model'] and config['augmentation']; accept
    # either 'aug' or 'augmentation' from --meta and fill sensible defaults.
    meta.setdefault("model", "yolov8n")
    if "augmentation" not in meta:
        meta["augmentation"] = meta.pop("aug", "?")
    report = {
        "experiment": args.name,
        "config": {"weights": str(weights), "imgsz": args.imgsz,
                   "trained_on": "kaggle/colab (offloaded)", "amp": S.AMP,
                   "conf_thresholds": S.CONF_THRESHOLDS, **meta},
        "dataset": {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                    "train_positives": split["counts"]["train_positives"],
                    "val_positives": split["counts"]["val_positives"],
                    "blackbox": {"tb": split["counts"]["test_positives"],
                                 "sick_non_tb": split["counts"]["blackbox_sick"],
                                 "healthy": split["counts"]["blackbox_healthy"]},
                    "class_map": {"0": "ActiveTuberculosis", "1": "ObsoletePulmonaryTuberculosis"}},
        "environment": environment(),
        "metrics": metric_block,
        "timing": {"eval_sec": eval_sec},
        "artifacts": {"weights": str(weights),
                      "confusion_matrix_test": str(out_dir / "confusion_matrix_test.png")},
    }
    metrics.save_confusion_png(metric_block["confusion_matrix"],
                               out_dir / "confusion_matrix_test.png")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "summary.txt").write_text(metrics.summary_text(report))

    print(f"\n→ {out_dir}/metrics.json")
    print(f"→ {out_dir}/summary.txt\n")
    print((out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
