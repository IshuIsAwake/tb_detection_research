"""
ap_compare.py — put YOLO and the torchvision detectors on EQUAL FOOTING.

THE PROBLEM THIS SOLVES
    RetinaNet's 0.739 Active mAP50 looked like it beat YOLO's matched-recipe
    0.708 (COCO-init, aug=geo, 512). Two confounds make that compare invalid:

    (a) DIFFERENT AP CODE. 0.708/0.745 are Ultralytics' AP; 0.739/0.726 are
        tv_detect.ap_from_per_image. The claimed +0.031 is the same size as the
        implementation gap — which was never measured.
    (b) DIFFERENT PREDICTION BUDGET. At the 0.001 conf floor, per image:
            YOLO geo 7.7  |  YOLO champion 17.9  |  FRCNN 18.4  |  RetinaNet 100.0
        RetinaNet is pinned at its detections_per_img cap on EVERY image — 13x
        YOLO's box count. 101-point interpolated AP takes the max precision to
        the right, so extra low-confidence true positives can only RAISE AP50.
        A bigger budget is free AP, and nothing in the headline controlled it.

WHAT IT DOES
    One AP implementation (tv_detect.ap_from_per_image) for every model, then a
    sweep of max_det caps applied identically to all of them. Inference runs
    ONCE per model and the raw preds are cached to JSON, so every later
    re-analysis (new caps, re-NMS, other AP settings) is free CPU work.

    Read the table down a CAP row, not across the uncapped one: at a matched
    budget, is RetinaNet still ahead of YOLO?

    `recall_ceil` (fraction of GT covered by ANY pred, conf ignored) says whether
    a cap deleted real detections or only noise — if it holds while mAP50 falls,
    the model had the lesion but ranked it below the cap.

CAVEATS (stated, not hidden)
    - Capping only moves DOWN. YOLO cannot be given more boxes after the fact
      (NMS already merged them), so the honest test is capping the torchvision
      models to YOLO's budget, not the reverse.
    - `--renms 0.5` optionally re-runs NMS at a common IoU (Ultralytics predicts
      at 0.7, torchvision at 0.5). Tighten-only, same reason.
    - Still single-seed, still our own split. This settles COMPARABILITY, not
      significance — that needs SEED=1/2 runs against the ±0.025 bar.

GPU RULE
    Inference is the user's to launch (121 images x 5 models). Smoke first:
        TB_SMOKE=1 python yolo_experiments/ap_compare.py --device cpu
    Then re-analysis without a GPU at all:
        python yolo_experiments/ap_compare.py --from-cache
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from yolo_common import metrics, settings as S, splits, tv_detect as T

R = S.RESULTS_ROOT

# name → (kind, weights, note). The five runs that are actually in play.
DEFAULT_MODELS = {
    "yolo_geo_coco":   ("yolo", R / "exp3_geo_512/ultralytics/weights/best.pt",
                        "YOLOv8n COCO-init, aug=geo — the matched-recipe anchor"),
    "yolo_geo_vindr":  ("yolo", S.REPO_ROOT / "cp_exp/results/exp3cp_geo_vindr_synth0_s0"
                                              "/ultralytics/weights/best.pt",
                        "YOLOv8n VinDr-init, aug=geo"),
    "yolo_champion":   ("yolo", R / "exp6_vindr_mosaic_mixup_fznone_s0/ultralytics/weights/best.pt",
                        "YOLOv8n VinDr-init, mosaic_mixup (separate arm) — seed 0 of 0.745+-0.028"),
    "retinanet":       ("retinanet", R / "retinanet_s0/best.pt",
                        "RetinaNet R50-FPNv2 COCO-init, aug=geo"),
    "fasterrcnn":      ("fasterrcnn", R / "fasterrcnn_s0/best.pt",
                        "Faster R-CNN R50-FPNv2 COCO-init, aug=geo"),
}


def parse_caps(spec: str) -> list[int | None]:
    return [None if c.strip().lower() in ("none", "off", "") else int(c)
            for c in spec.split(",")]


def parse_models(specs: list[str] | None) -> dict:
    if not specs:
        return DEFAULT_MODELS
    out = {}
    for s in specs:
        name, rest = s.split("=", 1)
        kind, path = rest.split(":", 1)
        out[name] = (kind, pathlib.Path(path), "user-supplied")
    return out


def stored_active_map50(weights: pathlib.Path) -> float | None:
    """The Active mAP50 the model's OWN run recorded — Ultralytics AP for a YOLO
    run, this AP code for a torchvision run. Walk up from the weights file to the
    run dir's metrics.json. Gives two things at once: the size of the AP-impl gap
    (YOLO rows) and a reproduction check (torchvision rows must land on 0.000)."""
    for up in range(4):
        p = pathlib.Path(weights).resolve().parents[up] / "metrics.json"
        if p.exists():
            try:
                return json.loads(p.read_text())["metrics"]["detection"]["active"]["mAP50"]
            except (KeyError, json.JSONDecodeError):
                return None
    return None


def md_table(rows: list[list], header: list[str]) -> str:
    """Markdown table, columns padded to the widest cell."""
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(header[i]), *(len(r[i]) for r in cells)) if cells else len(header[i])
         for i in range(len(header))]
    line = lambda cs: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cs)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (x + 2) for x in w) + "|",
                      *(line(c) for c in cells)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="override registry: name=kind:path (kind = yolo|retinanet|fasterrcnn)")
    ap.add_argument("--caps", default="none,100,30,20,10,5",
                    help="max_det budgets applied identically to every model")
    ap.add_argument("--renms", type=float, default=None,
                    help="re-run per-class NMS at this IoU for every model (tighten-only, e.g. 0.5)")
    ap.add_argument("--full-blackbox", action="store_true",
                    help="score all 360 sealed images (121 TB + 120 sick + 120 healthy) instead "
                         "of the TB positives only — adds the matched-sensitivity screening table. "
                         "AP is unchanged (it only ever reads the TB group)")
    ap.add_argument("--sens-targets", default="0.80,0.90,0.95",
                    help="sensitivities to match models at (WHO TPP triage bar = 0.90 sens / 0.70 spec)")
    ap.add_argument("--flag-class", choices=["any", "active"], default="any",
                    help="what counts as flagging an image: any lesion class (matches the existing "
                         "screening block) or ActiveTuberculosis only")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--from-cache", action="store_true",
                    help="skip inference entirely, re-analyse cached preds (no GPU)")
    ap.add_argument("--name", default="ap_compare")
    ap.add_argument("--cache-dir", default=None,
                    help="raw-pred cache (default <results>/<name>/preds) — point a variant run "
                         "at the primary run's cache so it costs no GPU and writes its own summary")
    args = ap.parse_args()

    models = parse_models(args.models)
    caps = parse_caps(args.caps)
    if args.full_blackbox and args.name == "ap_compare":
        args.name = "ap_compare_screen"   # 360-image preds must not collide with the TB-only cache
    if S.SMOKE and args.name.startswith("ap_compare"):
        args.name += "_smoke"          # never let 6-image smoke preds land in the real cache
    out_dir = R / args.name
    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else out_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = splits.build_or_load()
    if args.full_blackbox:
        items = splits.blackbox(split)
        if S.SMOKE:
            items = ([it for it in items if it[2] == "tb"][:6]
                     + [it for it in items if it[2] == "sick"][:4]
                     + [it for it in items if it[2] == "healthy"][:4])
        n_by_group = {g: sum(1 for it in items if it[2] == g) for g in ("tb", "sick", "healthy")}
        scope = f"{len(items)} sealed images {n_by_group}"
    else:
        items = T.tb_positive_items(split)
        if S.SMOKE:
            items = items[:6]
        scope = f"{len(items)} sealed TB positives"
    print(f"\n████ {args.name} — equal-footing AP  ({scope}, "
          f"imgsz={args.imgsz}, caps={args.caps}, renms={args.renms}, smoke={S.SMOKE}) ████\n")

    # ── 1. inference once per model (or load the cache) ────────────────────────
    preds = {}
    for name, (kind, path, note) in models.items():
        cache = cache_dir / f"{name}.json"
        if args.from_cache:
            if not cache.exists():
                print(f"  {name:16s} SKIP — no cache at {cache}")
                continue
            preds[name] = T.load_per_image(cache)
            print(f"  {name:16s} loaded {len(preds[name])} images from cache")
            continue
        if not pathlib.Path(path).exists():
            print(f"  {name:16s} SKIP — weights not found: {path}")
            continue
        t0 = time.time()
        preds[name] = T.per_image_for(kind, path, items, args.imgsz, args.device)
        T.dump_per_image(preds[name], cache)
        print(f"  {name:16s} {kind:11s} {len(items)} imgs in {time.time()-t0:5.1f}s  "
              f"→ {T.avg_preds(preds[name]):6.2f} preds/img  (cached)")
    if not preds:
        sys.exit("no models scored — check the weight paths above")

    # ── 2. same AP code, same budget, for everyone ────────────────────────────
    if args.renms is not None:
        preds = {n: T.renms_per_image(p, args.renms) for n, p in preds.items()}
        print(f"\n  re-NMS applied at IoU {args.renms} to every model (tighten-only)")

    grid = {}
    for name, per_image in preds.items():
        grid[name] = {}
        for cap in caps:
            capped = T.cap_per_image(per_image, cap)
            det = T.ap_from_per_image(capped)
            grid[name][str(cap)] = {
                "preds_per_img": T.avg_preds(capped),
                "active_mAP50": det["active"]["mAP50"],
                "active_mAP50_95": det["active"]["mAP50_95"],
                "active_P": det["active"]["precision"],
                "active_R": det["active"]["recall"],
                "obsolete_mAP50": det["obsolete"]["mAP50"],
                "overall_mAP50": det["overall"]["mAP50"],
                "active_recall_ceiling": T.recall_ceiling(capped, cls=0),
            }

    # ── 3. tables ─────────────────────────────────────────────────────────────
    names = list(grid)
    blocks = []
    for metric, label in [("active_mAP50", "Active mAP50 (same AP code, matched budget)"),
                          ("active_mAP50_95", "Active mAP50-95"),
                          ("active_recall_ceiling", "Active recall ceiling (conf ignored)"),
                          ("preds_per_img", "preds/img (the budget itself)")]:
        rows = [[("uncapped" if c is None else f"top-{c}")]
                + [f"{grid[n][str(c)][metric]:.3f}" for n in names] for c in caps]
        blocks.append(f"### {label}\n\n" + md_table(rows, ["cap"] + names))

    # ── matched-sensitivity screening (only possible with the negatives present) ──
    screen = {}
    if args.full_blackbox:
        targets = [float(t) for t in args.sens_targets.split(",")]
        cls = None if args.flag_class == "any" else 0
        screen = {n: metrics.matched_sensitivity(preds[n], targets, cls=cls) for n in names}
        rows = []
        for n in names:
            for t in targets:
                r = screen[n][f"{t:.2f}"]
                who = ("—" if t != 0.90 else
                       ("PASS" if r["specificity"] >= 0.70 and r["sensitivity"] >= 0.90 else "fail"))
                rows.append([n if t == targets[0] else "", f"{t:.2f}",
                             f"{r['threshold']:.4f}", r["tb_flagged"],
                             f"{r['specificity']:.3f}", f"{r['healthy_false_alarm']:.3f}",
                             f"{r['sick_false_alarm']:.3f}",
                             who if r["reachable"] else "unreachable"])
        blocks.insert(0, "### Matched sensitivity — each model at ITS OWN threshold\n\n"
                      + md_table(rows, ["model", "target sens", "threshold", "TB flagged",
                                        "specificity", "healthy FA", "sick FA", "WHO TPP"])
                      + "\n\n*Confidence is not a shared unit across architectures, so a fixed "
                        "conf 0.25 compares models at different sensitivities. Here sensitivity is "
                        "held fixed and specificity is read off. WHO TPP triage bar = >=0.90 "
                        "sensitivity AND >=0.70 specificity, judged on the 0.90 row.*")
        auc_rows = [[n, f"{screen[n]['auroc']['vs_all_negatives']:.4f}",
                     f"{screen[n]['auroc']['vs_healthy']:.4f}",
                     f"{screen[n]['auroc']['vs_sick_non_tb']:.4f}"] for n in names]
        blocks.insert(1, "### Image-level AUROC (threshold-free)\n\n"
                      + md_table(auc_rows, ["model", "TB vs all negatives", "vs healthy",
                                            "vs sick non-TB"])
                      + "\n\n*No operating point at all — the screening analogue of mAP. "
                        "`vs sick non-TB` is the hard one: other pathology mistaken for TB.*")

    # confound (a), measured: each run's own recorded number vs this AP code, uncapped.
    base = str(None) if None in caps else str(max(c for c in caps if c is not None))
    valid = (None in caps) and args.renms is None and not S.SMOKE
    gap_rows, gaps = [], {}
    for n in names:
        stored = stored_active_map50(models[n][1])
        mine = grid[n][base]["active_mAP50"]
        kind = "Ultralytics AP" if models[n][0] == "yolo" else "this AP code (repro check)"
        gaps[n] = {"as_run": stored, "this_ap_code": mine, "kind": kind,
                   "delta": None if stored is None else round(mine - stored, 4)}
        gap_rows.append([n, kind, "n/a" if stored is None else f"{stored:.3f}", f"{mine:.3f}",
                         "n/a" if stored is None else f"{mine - stored:+.3f}"])
    caption = ("*torchvision rows are a reproduction check — they were already scored with this "
               "code on these weights, so delta MUST be 0.000; anything else means this harness "
               "differs from the run. YOLO rows are the real AP-impl gap.*" if valid else
               "*⚠ NOT a valid comparison in this configuration (needs the uncapped row, "
               "--renms off, and the full image set) — smoke/re-NMS/cap change the predictions.*")
    gap_block = (f"### AP implementation gap (cap={base})\n\n"
                 + md_table(gap_rows, ["model", "as-run AP", "as run", "this AP code", "delta"])
                 + "\n\n" + caption)
    blocks.insert(0, gap_block)

    report = {
        "experiment": args.name,
        "ap_impl_gap": gaps,
        "config": {"imgsz": args.imgsz, "caps": [str(c) for c in caps], "renms": args.renms,
                   "n_images": len(items), "seed": S.SEED,
                   "ap_code": "tv_detect.ap_from_per_image (COCO-style 101-point) for ALL models",
                   "cap_semantics": "top-k by confidence per image, class-agnostic (= max_det)",
                   "note": "settles COMPARABILITY only — significance still needs seeds vs +-0.025"},
        "models": {n: {"kind": models[n][0], "weights": str(models[n][1]), "note": models[n][2]}
                   for n in names},
        "grid": grid,
    }
    if screen:
        report["config"]["flag_class"] = args.flag_class
        report["config"]["blackbox"] = {g: sum(1 for it in items if it[2] == g)
                                        for g in ("tb", "sick", "healthy")}
        report["screening_matched_sensitivity"] = screen
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    md = (f"# {args.name} — YOLO vs torchvision on equal footing\n\n"
          f"{scope} · one AP implementation for all models "
          f"· caps applied identically · re-NMS {args.renms}\n\n" + "\n\n".join(blocks) + "\n")
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"→ {out_dir}/metrics.json\n→ {out_dir}/summary.md\n→ {cache_dir}/ (raw preds; "
          f"re-analyse with --from-cache, no GPU)")


if __name__ == "__main__":
    main()
