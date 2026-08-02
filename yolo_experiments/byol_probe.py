"""
byol_probe.py — WHY did the SSL pretrain collapse? Four short arms, one verdict table.

THE FAILURE THIS EXISTS TO EXPLAIN
----------------------------------
The 2026-07-27 BYOL run on NIH ChestX-ray14 (112,120 imgs, ~8.8 h on Kaggle) produced a
100%-NaN backbone. Reading the log backwards:

    ep 1-2   loss 0.70 → 0.18   emb_std 0.034, 0.039   already ~half healthy
    ep 3-5   loss 0.089 → 0.019 emb_std 0.0045 → 0.0012  COLLAPSED (during LR warmup)
    ep 6-9   loss 0.11 → 0.53   emb_std ~0.003           thrashing
    ep 10-30 loss nan           emb_std nan              20 epochs / ~5.3 h wasted

For L2-normalised 256-d embeddings, healthy per-dim std ≈ 1/sqrt(256) = 0.0625. It was
never healthy — this was a slow collapse from epoch 1, NOT a gradient explosion. The NaN
is a downstream symptom ~7 epochs later. **So guards alone fix nothing** — they would
have reported the same failure at ep3 instead of hour 8.8. We need the CAUSE.

WHY THE OLD LOG COULDN'T ANSWER IT
----------------------------------
`emb_std` measures the very END of the tower (predictor output). It cannot separate "the
encoder went constant" from "the heads degenerated". `PROBE_LOG` in byol_pretrain.py adds
`z_enc_std` (the RAW encoder output, pre-projector) which localises the collapse, plus
`bn_var_min` (the smallest per-dim batch variance entering the projector's BatchNorm) which
tests the proposed NaN mechanism directly.

THE MEASURED MISMATCH (context for H1)
--------------------------------------
    yolov8n backbone (encoder):   1,272,656 params   feature dim  256
    BYOL projector + predictor:   2,105,856 params
    reference BYOL (ResNet-50):  25,557,032 params   feature dim 2048

The BYOL heads are 1.66x LARGER than the encoder they wrap, and every hyperparameter in
byol_pretrain.py (LR 1e-3, tau 0.996, 2048-hidden proj, 5-epoch warmup) came from the
ResNet-50 recipe.

THE ARMS
--------
    A0  TARGET_BN_MODE=eval          MUST reproduce the collapse, or the probe is invalid.
                                          This WAS the shipped default; since 2026-07-28 the
                                          script defaults to `train`, so A0 pins eval itself.
    A1  TARGET_BN_MODE=train          H2: reference BYOL keeps the target in TRAIN mode so
                                          its BN uses BATCH stats — the mechanism BYOL's
                                          collapse resistance is attributed to. We shipped
                                          eval() + copied running stats. ~4-line deviation.
                                          CONFIRMED as the cause → now the default.
    A2  LR=3e-4 GRAD_CLIP=1.0         H3: LR 1e-3 at batch 64 with NO gradient clipping.
    A3  OBJECTIVE=vicreg              H1: BYOL resists collapse only IMPLICITLY. VICReg
                                          adds an EXPLICIT variance hinge penalising the
                                          exact failure measured (per-dim std -> 0), while
                                          staying non-contrastive and negative-free.

DECISION RULE (fixed BEFORE the numbers — ranked by novelty of the outcome)
---------------------------------------------------------------------------
  1. A1 holds            -> real BYOL works on yolov8n; the collapse was a 4-line bug.
                            Strongest claim available. Relaunch 112k with TARGET_BN_MODE=train.
  2. A3 holds (A1 not)   -> VICReg on yolov8n. Still non-contrastive, still novel on a YOLO
                            backbone. Relaunch 112k with OBJECTIVE=vicreg.
  3. A2 only             -> hyperparameter fragility; retune, but treat as brittle.
  4. nothing holds       -> yolov8n is not viable as a non-contrastive SSL encoder at this
                            scale. Fall back to SSL on ResNet-50 -> the RetinaNet already in
                            yolo_common/tv_detect.py (0.739, best recall ceiling 0.972).

WHY SimCLR IS NOT AN ARM (decided 2026-07-27)
---------------------------------------------
The YOLO-backbone SSL literature (Self-Supervised YOLO, arXiv 2508.01966; SSL-YOLO; the
Lightly tutorial) uses SimCLR, and no source uses BYOL on a YOLO backbone. It still does not
fit THIS data:
  - Contrastive "negatives" are OTHER IMAGES IN THE BATCH, not disease-negative patients.
    NIH ChestX-ray14 is 60,361 "No Finding" vs 51,759 with >=1 finding (= 112,120; ~54%
    normal), so a batch of 64 holds ~34 anatomically near-identical normal CXRs that
    SimCLR's loss explicitly pushes APART. That is the false-negative problem: contrastive
    SSL is actively wrong on a homogeneous corpus.
  - SimCLR's load-bearing augmentation is COLOUR JITTER, which is meaningless on grayscale
    CXR (the MoCo-CXR lesson already cited in byol_pretrain.py). BYOL-family methods degrade
    far more gracefully under a reduced augmentation set.
That literature is COCO/natural images, where negatives are semantically valid and colour
jitter works. Neither property transfers.

RESOLUTION CAVEAT
-----------------
Arms run at SSL_IMGSZ=256 for speed, assuming the collapse mechanism is not
resolution-dependent. **A0 reproducing the collapse at 256 is what validates that
assumption.** If A0 does NOT collapse, re-run A0 at 512 (--imgsz 512) before believing any
other arm.

USAGE (the USER runs this — GPU rule)
-------------------------------------
    python yolo_experiments/byol_probe.py                  # all four arms, ~40 min total
    python yolo_experiments/byol_probe.py --arms A0 A1     # just the BYOL-bug question
    python yolo_experiments/byol_probe.py --steps 200      # faster / rougher

Corpus defaults to data/TBX11K/imgs/test (3302 imgs) — leakage-safe for SSL: it is OUTSIDE
the frozen split and carries no labels we score against. SSL is label-free, so the mixed
healthy/sick/TB composition is a non-issue (it only matters for pseudo-labelling, exp12).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRETRAIN = REPO / "kaggle" / "byol_pretrain.py"
DEFAULT_CORPUS = REPO / "data" / "TBX11K" / "imgs" / "test"
OUT_DIR = REPO / "yolo_experiments" / "results" / "byol_probe"

# arm -> (label, extra env). Everything else is held identical across arms.
#
# ⚠ EVERY ARM PINS TARGET_BN_MODE EXPLICITLY — none of them inherit it.
# The arms used to ride byol_pretrain.py's default (A0 was an empty dict), which was
# fine while that default was `eval`. It moved to `train` on 2026-07-28 once this very
# probe named `eval` as the collapse. Inheriting would have silently changed the
# experiment in two ways: A0 would become a duplicate of A1 and stop reproducing the
# collapse it exists to reproduce, and A2/A3 would quietly become "target-BN fix PLUS
# my deviation" instead of single-deviation arms — so A2 would no longer be a test of
# H3 at all, and the recorded "H3 rejected" would look wrong on a re-run.
# The design is: A0 = the failing baseline, A1/A2/A3 = that baseline plus ONE change.
BASELINE_ENV = {"TARGET_BN_MODE": "eval"}          # the configuration that collapsed
ARMS: dict[str, tuple[str, dict[str, str]]] = {
    "A0": ("baseline: target BN eval (must reproduce collapse)", {**BASELINE_ENV}),
    "A1": ("target BN in train mode  [H2]", {**BASELINE_ENV, "TARGET_BN_MODE": "train"}),
    "A2": ("LR 3e-4 + grad clip 1.0  [H3]", {**BASELINE_ENV, "LR": "3e-4", "GRAD_CLIP": "1.0"}),
    "A3": ("VICReg objective         [H1]", {**BASELINE_ENV, "OBJECTIVE": "vicreg"}),
}

REF_RE = re.compile(r"\[ref\].*?emb_std=([0-9.eE+-]+)\s+z_enc_std=([0-9.eE+-]+)")
PROBE_RE = re.compile(
    r"\[probe ep(\d+) step(\d+)\]\s+loss=(\S+)\s+z_enc_std=(\S+)\s+\(ref\s+(\S+)\)\s+"
    r"bn_var_min=(\S+)\s+z_proj_std=(\S+)\s+p_std=(\S+)\s+g_enc=(\S+)\s+scale=(\S+)")


def run_arm(arm: str, args) -> dict:
    """One arm in its OWN process. byol_pretrain.py reads config at IMPORT time (the
    documented tbx_train landmine), so an in-process loop would silently reuse arm A0's
    settings for every later arm."""
    label, extra = ARMS[arm]
    env = os.environ.copy()
    env.update({
        "INPUT_ROOT": str(args.corpus),
        "WORK_ROOT": str(OUT_DIR / arm),
        "OUT_NAME": f"probe_{arm}",
        "SSL_IMGSZ": str(args.imgsz),
        "BATCH": str(args.batch),          # MUST match the real run — batch size is part of H1
        # EPOCHS must be big enough that MAX_STEPS is what actually stops the arm. Setting it
        # to 1 capped every arm at len(loader) steps (51 on this corpus) and silently produced
        # a probe 100x too short to see a collapse that took ~5250 steps in the real run.
        "EPOCHS": "10000",
        "MAX_STEPS": str(args.steps),      # also drives the LR/tau cosine (see byol_pretrain)
        "WARMUP_STEPS": str(max(1, args.steps // 6)),   # mirrors the real 5-of-30-epoch warmup
        "PROBE_LOG": str(args.log_every),
        "COLLAPSE_HALT": "0",              # we WANT to watch it collapse, not halt on it
        "WORKERS": str(args.workers),
        "GPU_AUG": "1", "BLUR_MAX": "9", "PRECACHE": "0",
        "SAVE_EVERY": "0", "MULTI_GPU": "0", "DEVICE": args.device,
        "SEED": "0",
    })
    env.update(extra)
    (OUT_DIR / arm).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}\n▶ {arm}: {label}\n  {extra or 'no overrides'}\n{'=' * 78}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(PRETRAIN)], env=env,
                          capture_output=True, text=True)
    dt = time.time() - t0
    log = proc.stdout + proc.stderr
    (OUT_DIR / arm / "log.txt").write_text(log)

    rows = [m.groups() for m in PROBE_RE.finditer(log)]
    ref = REF_RE.search(log)
    res = {
        "arm": arm, "label": label, "extra": extra, "seconds": round(dt, 1),
        "returncode": proc.returncode,
        "ref_emb_std": float(ref.group(1)) if ref else None,
        "ref_enc_std": float(ref.group(2)) if ref else None,
        "trace": [{"step": int(r[1]), "loss": float(r[2]), "z_enc_std": float(r[3]),
                   "bn_var_min": float(r[5]), "z_proj_std": float(r[6]),
                   "p_std": float(r[7]), "g_enc": float(r[8]), "scale": float(r[9])}
                  for r in rows],
    }
    if proc.returncode != 0:
        tail = [ln for ln in log.strip().splitlines() if ln.strip()][-3:]
        print(f"  ⚠ exit {proc.returncode}: " + " | ".join(tail), flush=True)
    print(f"  {len(res['trace'])} probe points in {dt:.0f}s", flush=True)
    return res


def verdict(r: dict) -> str:
    """Held / collapsed, judged on p_std — the L2-NORMALISED embedding spread.

    ⚠ THIS WAS WRONG ONCE, and the mistake is instructive. The first version judged on
    z_enc_std (the RAW encoder output) and reported the as-shipped arm as "HELD (1174%)"
    while it was in fact collapsing to 1% — because **the collapse is DIRECTIONAL, not a
    shrink**. The encoder learns c(x)·v: a fixed direction v with a per-image magnitude
    c(x). Raw per-dim std across the batch therefore EXPLODES (magnitudes vary), while
    every L2-normalised embedding points the same way, which is what the loss actually
    sees and what the detector would inherit. So p_std is the metric; a z_enc_std blow-up
    is a second, independent symptom worth flagging, not a sign of health.
    """
    if not r["trace"] or not r.get("ref_emb_std"):
        return "NO DATA"
    last = r["trace"][-1]
    if any(v != v for v in (last["p_std"], last["loss"], last["z_enc_std"])):   # NaN
        return "NaN"
    frac = last["p_std"] / r["ref_emb_std"]
    enc_blowup = (r["ref_enc_std"] or 0) and last["z_enc_std"] / r["ref_enc_std"] > 4.0
    tag = f"{frac:.0%}" + (" +enc-blowup" if enc_blowup else "")
    if frac < 0.25:
        return f"COLLAPSED ({tag})"
    if frac < 0.60:
        return f"degrading ({tag})"
    return f"HELD ({tag})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the BYOL collapse. See module docstring.")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    # The real run collapsed at ~epoch 3 of 30 = ~5250 optimizer steps. A probe far short of
    # that cannot reproduce it, so the step budget is the thing that makes this valid — at
    # ~0.16 s/step (imgsz 256, batch 64, RTX 3050) 3000 steps is ~8 min per arm.
    ap.add_argument("--steps", type=int, default=3000, help="steps per arm (~8 min each)")
    ap.add_argument("--imgsz", type=int, default=256, help="256 for speed; see RESOLUTION CAVEAT")
    ap.add_argument("--batch", type=int, default=64, help="MUST match the real run")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--report", action="store_true",
                    help="re-print the verdict from the saved probe.json (no GPU, no re-run)")
    args = ap.parse_args()

    if args.report:
        pass
    elif not pathlib.Path(args.corpus).is_dir():
        raise SystemExit(f"corpus not found: {args.corpus}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.report:                      # re-score an existing run; free, no GPU
        results = json.loads((OUT_DIR / "probe.json").read_text())
    else:
        results = [run_arm(a, args) for a in args.arms]
        (OUT_DIR / "probe.json").write_text(json.dumps(results, indent=2))

    print(f"\n\n{'=' * 78}\nVERDICT — p_std (L2-normalised embedding spread) vs its step-0 reference"
          f"\n{'=' * 78}")
    print(f"{'arm':<4} {'config':<36} {'steps':>6} {'p_std':>8} {'z_enc':>8} "
          f"{'bn_var_min':>11} {'loss':>9}  verdict")
    print("-" * 104)
    for r in results:
        last = r["trace"][-1] if r["trace"] else {}
        print(f"{r['arm']:<4} {r['label']:<36} {last.get('step', 0):>6} "
              f"{last.get('p_std', float('nan')):>8.5f} {last.get('z_enc_std', float('nan')):>8.4f} "
              f"{last.get('bn_var_min', float('nan')):>11.2e} {last.get('loss', float('nan')):>9.4f}"
              f"  {verdict(r)}")

    by = {r["arm"]: verdict(r) for r in results}
    short = [r["arm"] for r in results
             if r["trace"] and r["trace"][-1]["step"] < 0.9 * args.steps and r["returncode"] == 0]
    print("\nREAD IT WITH THE DECISION RULE (module docstring):")
    if short:
        print(f"  ⚠ {', '.join(short)} stopped WELL SHORT of --steps {args.steps} → the arm ended "
              f"early (epoch budget? crash?). A probe shorter than the ~5250 steps the real run "
              f"took to collapse proves NOTHING. Check the arm logs before reading anything below.")
    if "A0" in by and not by["A0"].startswith(("COLLAPSED", "NaN", "degrading")):
        print(f"  ⚠ A0 did NOT reproduce the collapse in {args.steps} steps → the probe cannot "
              f"settle the question yet. Try --steps {args.steps * 2}, and only then --imgsz 512.")
    else:
        for arm, want in (("A1", "real BYOL is salvageable (4-line target-BN fix)"),
                          ("A3", "VICReg on yolov8n — the non-contrastive fallback"),
                          ("A2", "hyperparameter fragility only")):
            if arm in by and by[arm].startswith("HELD"):
                print(f"  → {arm} HELD: {want}")
                break
        else:
            print("  → nothing held. yolov8n may not be viable as a non-contrastive SSL "
                  "encoder at this scale; see fallback 4 in the docstring.")
    print(f"\nfull traces: {OUT_DIR / 'probe.json'}")


if __name__ == "__main__":
    main()
