"""
collect_cv.py — gather the rotating-fold CV runs into one table under conclusion/.

WHAT THIS READS
    yolo_experiments/results/retinanet_cv_<config>_fold<i>/metrics.json

    Those are the runs produced by `retinanet.py --fold i` (see tv_detect.py
    _resolve_split). Each one was scored on ONE fold of the k-way partition of
    all ~799 TB positives, so across the k runs of a config every positive is
    tested exactly once — the whole positive set is the test set, not the sealed
    121.

WHAT IT WRITES
    conclusion/cv_results.json   machine-readable: every fold + per-config stats
    conclusion/cv_results.md     the table to paste into the report

WHY IT LIVES HERE AND NOT IN yolo_experiments/
    conclusion/ holds the numbers that go in the write-up. Run outputs stay
    where they are; this only ever READS them.

⚠ THREE THINGS THAT MAKE A NUMBER FROM HERE WRONG IF YOU IGNORE THEM
  1. A fold number is NOT comparable to a sealed-360 cell. Different test set,
     different size. Never put them in the same table.
  2. Fold test sets are POSITIVES-ONLY, so screening / false-alarm rows are zero
     by construction. This script refuses to report them at all.
  3. Single-class runs report under the key `lesion` on 179 merged GT boxes, NOT
     the 142-box Active number. The key is recorded per row so the two can never
     be silently pooled.

USAGE
    python conclusion/collect_cv.py                 # all retinanet_cv_* runs
    python conclusion/collect_cv.py --pattern 'retinanet_cv_*'
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import settings as S

HERE = Path(__file__).resolve().parent
RUN_RE = re.compile(r"^retinanet_cv_(?P<config>.+)_fold(?P<fold>\d+)$")


def load_runs(pattern: str) -> list[dict]:
    """One row per finished fold run. Silently skips runs still in flight."""
    rows = []
    for d in sorted(S.RESULTS_ROOT.glob(pattern)):
        m = RUN_RE.match(d.name)
        mj = d / "metrics.json"
        if not m or not mj.exists():
            continue
        r = json.loads(mj.read_text())
        kf = (r.get("dataset") or {}).get("kfold")
        if not kf:
            # A run whose name matches but which was scored on the sealed split
            # would silently corrupt the table — refuse it rather than average it in.
            print(f"  ⚠ SKIP {d.name}: no `kfold` block — this was NOT a fold run")
            continue
        det = r["metrics"]["detection"]
        key = "lesion" if "lesion" in det else "active"
        rows.append({
            "run": d.name, "config": m["config"], "fold": int(m["fold"]),
            "k": kf["k"], "key": key,
            "mAP50": det[key]["mAP50"], "mAP50_95": det[key]["mAP50_95"],
            "n_test": int(kf["fold_sizes"][int(m["fold"])]),
            "train_positives": r["dataset"]["train_positives"],
            "epochs_ran": r["config"]["epochs_ran"],
            "best_epoch": r["config"]["best_epoch"],
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    """mean / sd / n per config. sd is the SPREAD ACROSS FOLDS — it is not the
    seed noise (σ≈0.010) and not the ±0.025 significance bar. Fold-to-fold
    spread mixes split difficulty with run noise; that is what makes it the
    honest error bar for 'how well does this config do on THIS dataset'."""
    out = {}
    for cfg in sorted({r["config"] for r in rows}):
        vals = [r["mAP50"] for r in rows if r["config"] == cfg]
        folds = sorted(r["fold"] for r in rows if r["config"] == cfg)
        out[cfg] = {
            "n_folds": len(vals), "folds": folds,
            "mean_mAP50": round(statistics.mean(vals), 4),
            "sd_mAP50": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "n_test_total": sum(r["n_test"] for r in rows if r["config"] == cfg),
        }
    return out


def paired_deltas(rows: list[dict]) -> list[dict]:
    """Per-fold differences between configs — the whole point of running every
    config on the SAME folds. A paired delta cancels 'this fold happens to be
    hard', which is the dominant term in the unpaired spread."""
    by = {(r["config"], r["fold"]): r["mAP50"] for r in rows}
    cfgs = sorted({r["config"] for r in rows})
    out = []
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            shared = sorted({f for (c, f) in by if c == a} & {f for (c, f) in by if c == b})
            if not shared:
                continue
            d = [by[(b, f)] - by[(a, f)] for f in shared]
            out.append({
                "a": a, "b": b, "folds": shared, "per_fold": [round(x, 4) for x in d],
                "mean_delta": round(statistics.mean(d), 4),
                "sd_delta": round(statistics.stdev(d), 4) if len(d) > 1 else None,
                # Sign agreement is the cheap robustness read: a real effect
                # points the same way on every fold. 3/5 is a coin flip no
                # matter how big the mean looks.
                "folds_favouring_b": sum(1 for x in d if x > 0), "n_folds": len(d),
            })
    return out


def md_table(rows: list[list[str]], head: list[str]) -> str:
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
                     + [line(r) for r in rows])


def render(rows, stats, deltas) -> str:
    keys = sorted({r["key"] for r in rows})
    L = ["# Cross-validation — rotating disjoint test folds", "",
         f"Metric key: **{', '.join(keys)}** "
         f"({'179 merged lesion boxes' if keys == ['lesion'] else 'see per-row key'}). "
         f"k={rows[0]['k']}, every TB positive tested exactly once across the folds.", "",
         "⚠ These are NOT comparable to sealed-360 numbers — different test set. "
         "Screening / false-alarm rows do not exist here (folds are positives-only).", ""]

    L += ["## Per-config", "",
          md_table([[c, s["n_folds"],
                     f"{s['mean_mAP50']:.4f}" + (f" ± {s['sd_mAP50']:.4f}" if s["sd_mAP50"] else ""),
                     f"{s['min']:.3f}–{s['max']:.3f}", s["n_test_total"]]
                    for c, s in stats.items()],
                   ["config", "folds", "mean mAP50", "range", "test images"]), ""]

    if deltas:
        L += ["## Paired deltas (same folds, so fold difficulty cancels)", "",
              md_table([[f"{d['b']} − {d['a']}",
                         f"{d['mean_delta']:+.4f}" + (f" ± {d['sd_delta']:.4f}" if d["sd_delta"] else ""),
                         f"{d['folds_favouring_b']}/{d['n_folds']}",
                         ", ".join(f"{x:+.3f}" for x in d["per_fold"])]
                        for d in deltas],
                       ["contrast", "mean Δ", "folds favouring", "per fold"]), "",
              "**Read the sign-agreement column first.** A real effect points the same "
              "way on every fold; a split like 3/5 is a coin flip however large the mean. "
              "The arm's measured seed noise is σ ≈ 0.010 and its significance bar is "
              "±0.025 — both derived on the SEALED split, so treat them as a rough guide "
              "here, not a test.", ""]

    L += ["## Every fold", "",
          md_table([[r["config"], r["fold"], f"{r['mAP50']:.4f}", f"{r['mAP50_95']:.4f}",
                     r["n_test"], r["train_positives"], f"{r['best_epoch']}/{r['epochs_ran']}"]
                    for r in sorted(rows, key=lambda x: (x["config"], x["fold"]))],
                   ["config", "fold", "mAP50", "mAP50-95", "test", "train", "best/ran"]), ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pattern", default="retinanet_cv_*")
    args = ap.parse_args()

    rows = load_runs(args.pattern)
    if not rows:
        raise SystemExit(f"no finished fold runs matching {args.pattern!r} under "
                         f"{S.RESULTS_ROOT} — nothing to collect yet")
    stats, deltas = summarise(rows), paired_deltas(rows)

    incomplete = [c for c, s in stats.items() if s["n_folds"] < rows[0]["k"]]
    if incomplete:
        print(f"  ⚠ PARTIAL — {', '.join(incomplete)} have fewer than k={rows[0]['k']} "
              f"folds. Means below are over the folds present, so they are NOT yet "
              f"'every positive tested once'.")

    (HERE / "cv_results.json").write_text(json.dumps(
        {"runs": rows, "per_config": stats, "paired_deltas": deltas}, indent=2))
    (HERE / "cv_results.md").write_text(render(rows, stats, deltas))
    print(f"  {len(rows)} fold runs, {len(stats)} configs")
    print(f"→ {HERE / 'cv_results.json'}\n→ {HERE / 'cv_results.md'}\n")
    print((HERE / "cv_results.md").read_text())


if __name__ == "__main__":
    main()
