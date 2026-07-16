"""
exp9_cross_domain.py — Cross-domain generalisation of the locked detector.

The YOLO detector was trained + evaluated entirely on TBX11K's own `tb/` images.
This asks the honest external question: does it still fire on TB lesions on
chest X-rays from *different* hospitals/scanners it never saw?

External images: TBX11K bundles a train/val slice of Montgomery (MCUCXR_*) and
Shenzhen (CHNCXR_*) under `imgs/extra/mc+shenzhen/`, already downsampled to 512²
by the TBX11K pipeline — so preprocessing matches training, and (verified) NONE
of these ids appear in our frozen split, so there is no training leakage. They
carry image-level TB/normal labels only (filename suffix `_1`=TB, `_0`=normal),
NO lesion boxes — so this is an image-level SENSITIVITY test + a box-quality
eyeball (overlays are a separate script), not a mAP.

We compare four inference configs, each over 3 seeds (mean ± std):

    best_512        VinDr + mosaic_mixup + full FT @ 512   (the champion)
    best_512_tta    same weights, test-time augmentation
    best_1024       VinDr + mosaic        + full FT @ 1024  (exp7 1024 best)
    best_1024_tta   same weights, test-time augmentation

A model "fires" on an image if it emits ≥1 box at conf ≥ threshold. We report,
per conf threshold and per source:
  - sensitivity (TB-only):  fired_TB / TB_count          ← the number we care about
  - screening yield (all):  fired_TB / all_images         ← TB fires over the whole pool
  - false-alarm (normal):   fired_normal / normal_count   ← diagnostic (positives-only
                            detector is expected to fire on abnormal-but-non-TB lungs)

Inference only — no training. Runs locally (fits 6 GB even at 1024). Predictions
are cached to predictions.json so the overlay script does not re-run inference.

    python yolo_experiments/exp9_cross_domain.py            # all 4 configs, 3 seeds
    python yolo_experiments/exp9_cross_domain.py --seeds 0  # quick single-seed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import settings as S, train_eval

# ── External image set (local, preprocessing-matched, no leakage) ─────────────
EXTRA_DIR = S.IMGS_ROOT / "extra" / "mc+shenzhen"   # train/ + val/ subdirs

# ── The four configs. weights are per-seed templates relative to REPO_ROOT. ───
CONFIGS = {
    "best_512": {
        "weights": "yolo_experiments/results/exp6_vindr_mosaic_mixup_fznone_s{seed}/ultralytics/weights/best.pt",
        "imgsz": 512, "augment": False,
        "recipe": "VinDr + mosaic_mixup + full FT @ 512",
    },
    "best_512_tta": {
        "weights": "yolo_experiments/results/exp6_vindr_mosaic_mixup_fznone_s{seed}/ultralytics/weights/best.pt",
        "imgsz": 512, "augment": True,
        "recipe": "VinDr + mosaic_mixup + full FT @ 512, +TTA",
    },
    "best_1024": {
        "weights": "weights/tbx11k/1024_batch_16/1024_mosaic_s{seed}.pt",
        "imgsz": 1024, "augment": False,
        "recipe": "VinDr + mosaic + full FT @ 1024",
    },
    "best_1024_tta": {
        "weights": "weights/tbx11k/1024_batch_16/1024_mosaic_s{seed}.pt",
        "imgsz": 1024, "augment": True,
        "recipe": "VinDr + mosaic + full FT @ 1024, +TTA",
    },
}

STORE_CONF = 0.01   # keep boxes >= this per image (enough for all thresholds + overlay)


def discover() -> list[dict]:
    """All external images with parsed source + image-level label."""
    items = []
    for p in sorted(EXTRA_DIR.rglob("*.png")):
        name = p.stem                       # e.g. CHNCXR_0347_1
        source = "Montgomery" if name.startswith("MCUCXR") else "Shenzhen"
        label = "tb" if name.endswith("_1") else "normal"
        items.append({"path": p, "name": name, "source": source, "label": label})
    return items


def predict_one_config(weights: Path, imgsz: int, augment: bool, items: list[dict]) -> dict:
    """Run one model over all images; return {name: {..., boxes:[[x1,y1,x2,y2,conf]]}}."""
    model = train_eval.load_best(weights)
    out = {}
    for it in items:
        res = model.predict(str(it["path"]), imgsz=imgsz, device=S.DEVICE,
                            conf=S.RECALL_CEILING_CONF, verbose=False,
                            augment=augment)[0]
        boxes = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                if c >= STORE_CONF:
                    boxes.append([round(float(x1), 1), round(float(y1), 1),
                                  round(float(x2), 1), round(float(y2), 1),
                                  round(float(c), 4), int(k)])
        boxes.sort(key=lambda b: -b[4])
        out[it["name"]] = {"source": it["source"], "label": it["label"],
                           "max_conf": (boxes[0][4] if boxes else 0.0),
                           "boxes": boxes}
    return out


def fire_rates(preds: dict, conf: float) -> dict:
    """Per-source and combined fire counts at a conf threshold."""
    # buckets[source][label] = [n_fired, n_total]
    from collections import defaultdict
    b = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for rec in preds.values():
        src, lab = rec["source"], rec["label"]
        fired = int(rec["max_conf"] >= conf)
        b[src][lab][0] += fired
        b[src][lab][1] += 1
        b["Combined"][lab][0] += fired
        b["Combined"][lab][1] += 1

    def block(src):
        tb_f, tb_n = b[src]["tb"]
        nm_f, nm_n = b[src]["normal"]
        alln = tb_n + nm_n
        return {
            "tb_fired": tb_f, "tb_count": tb_n,
            "normal_fired": nm_f, "normal_count": nm_n,
            "sensitivity_tb_only": (tb_f / tb_n) if tb_n else None,
            "sensitivity_all_denom": (tb_f / alln) if alln else None,  # TB fires / all imgs
            "normal_false_alarm": (nm_f / nm_n) if nm_n else None,
        }
    return {src: block(src) for src in ("Montgomery", "Shenzhen", "Combined")}


def aggregate_seeds(per_seed_rates: list[dict], conf: float) -> dict:
    """mean ± std across seeds of the key rates, per source."""
    srcs = ("Montgomery", "Shenzhen", "Combined")
    keys = ("sensitivity_tb_only", "sensitivity_all_denom", "normal_false_alarm")
    agg = {}
    for src in srcs:
        agg[src] = {}
        for k in keys:
            vals = [r[src][k] for r in per_seed_rates if r[src][k] is not None]
            agg[src][k] = {"mean": round(mean(vals), 4) if vals else None,
                           "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
                           "per_seed": [round(v, 4) for v in vals]}
        # counts are identical across seeds; take from the first
        agg[src]["counts"] = {kk: per_seed_rates[0][src][kk]
                              for kk in ("tb_count", "normal_count")}
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2", help="comma seeds (default 0,1,2)")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma config names (default all four)")
    ap.add_argument("--name", default="exp9_cross_domain", help="results/<name>/")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cfg_names = [c.strip() for c in args.configs.split(",")]

    items = discover()
    src_counts = {}
    for it in items:
        src_counts.setdefault(it["source"], {"tb": 0, "normal": 0})
        src_counts[it["source"]][it["label"]] += 1
    print(f"\n████ exp9 cross-domain — {len(items)} external images ████")
    for s, c in src_counts.items():
        print(f"  {s}: {c['tb']} TB + {c['normal']} normal")
    print(f"  configs={cfg_names}  seeds={seeds}\n")

    out_dir = S.RESULTS_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_preds = {}          # config -> seed -> preds  (cached for overlay script)
    results = {}            # config -> aggregated rates per conf
    t0 = time.time()

    for cfg_name in cfg_names:
        cfg = CONFIGS[cfg_name]
        all_preds[cfg_name] = {}
        per_seed_rates_by_conf = {c: [] for c in S.CONF_THRESHOLDS}
        for seed in seeds:
            w = S.REPO_ROOT / cfg["weights"].format(seed=seed)
            if not w.exists():
                print(f"  [skip] {cfg_name} s{seed}: weights missing ({w})")
                continue
            tag = "TTA" if cfg["augment"] else "   "
            print(f"  {cfg_name} s{seed} [{tag}] imgsz={cfg['imgsz']} … ", end="", flush=True)
            ts = time.time()
            preds = predict_one_config(w, cfg["imgsz"], cfg["augment"], items)
            all_preds[cfg_name][str(seed)] = preds
            for conf in S.CONF_THRESHOLDS:
                per_seed_rates_by_conf[conf].append(fire_rates(preds, conf))
            print(f"{time.time()-ts:.1f}s")
        results[cfg_name] = {
            "recipe": cfg["recipe"], "imgsz": cfg["imgsz"], "augment": cfg["augment"],
            "seeds_used": [s for s in seeds
                           if (S.REPO_ROOT / cfg["weights"].format(seed=s)).exists()],
            "by_conf": {f"{conf:.2f}": aggregate_seeds(per_seed_rates_by_conf[conf], conf)
                        for conf in S.CONF_THRESHOLDS
                        if per_seed_rates_by_conf[conf]},
        }

    report = {
        "experiment": args.name,
        "kind": "cross-domain (Montgomery + Shenzhen, external, image-level)",
        "source": str(EXTRA_DIR),
        "note": ("image-level sensitivity only (no external lesion boxes). "
                 "positives-only detector expected to fire on abnormal non-TB lungs."),
        "counts": src_counts,
        "conf_thresholds": S.CONF_THRESHOLDS,
        "environment": {"timestamp": datetime.now(timezone.utc).isoformat(),
                        "device": S.DEVICE},
        "results": results,
        "timing_sec": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps(all_preds))
    (out_dir / "summary.txt").write_text(summary_text(report))
    print(f"\n→ {out_dir}/metrics.json")
    print(f"→ {out_dir}/predictions.json  (cached for overlay)")
    print(f"→ {out_dir}/summary.txt\n")
    print((out_dir / "summary.txt").read_text())


def summary_text(report: dict) -> str:
    L = [f"# {report['experiment']} — {report['kind']}",
         f"# {report['note']}", ""]
    for src, c in report["counts"].items():
        L.append(f"#   {src}: {c['tb']} TB + {c['normal']} normal")
    L.append("")
    for cfg_name, r in report["results"].items():
        L.append(f"── {cfg_name}  [{r['recipe']}]  seeds={r['seeds_used']}")
        for conf, agg in r["by_conf"].items():
            L.append(f"   conf≥{conf}:")
            for src in ("Montgomery", "Shenzhen", "Combined"):
                a = agg[src]
                st = a["sensitivity_tb_only"]; sa = a["sensitivity_all_denom"]
                fa = a["normal_false_alarm"]; ct = a["counts"]
                def fmt(d):
                    return f"{d['mean']:.3f}±{d['std']:.3f}" if d["mean"] is not None else "  n/a "
                L.append(f"     {src:11s} sens(TB {ct['tb_count']})={fmt(st)} "
                         f"| sens(all)={fmt(sa)} "
                         f"| FA(norm {ct['normal_count']})={fmt(fa)}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
