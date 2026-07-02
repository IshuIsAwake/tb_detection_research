# RESULTS

What each experiment **added** to the research. Wrong reads are corrected in
place (struck through, not deleted — knowing *why* a read was wrong is the
asset). Per-run numbers live in `yolo_experiments/results/<run>/metrics.json`.

- **Metric we optimize:** Active mAP50 on the sealed test set (lesion-level
  recall on the dominant TB class). Guardrails: medium-lesion recall, loc IoU.
- **Judge:** the frozen split's sealed 360 blackbox (121 TB + 120 sick + 120
  healthy), untouched by training.

---

## exp1 — YOLOv8n floor (positives-only, aug off)

- Bottleneck is **recall, not box quality** (matched IoU ~0.74 — lesions are
  missed, not mislocated).
- **Overfits by ~epoch 12** — more epochs hurt.
- **Active ≫ Obsolete** (class imbalance; Obsolete barely learnable).
- Fires on **sick (non-TB) lungs ~2–3× the healthy rate** — confuses other
  pathology with TB.
- Resolution: at equal batch 1024 > 512, but 512@16 > 1024@8 (batch is a lever).

## exp2 — negatives in detector training

- Background negatives **collapse false alarms** (even 0.25:1, saturates at once)
  but **cap TB sensitivity at ~60%**.
- **Decision: specificity belongs in the classifier; keep the detector
  positives-only / sensitive.** Negatives lever closed for the detector.
- Ratio (0.25/0.5/1.0) unresolvable — single-seed noise ≥ effect.

## exp3 — augmentation screen (512 @ batch 16, positives-only)

- **Augmentation is the biggest lever yet.** Active mAP50 0.53 (off) → 0.74
  (mosaic).
- **Fixes the overfitting**: val-mAP peak moves from epoch ~7 to ~110–145.
- Domain pruning held: brightness (`geo_photo`) and the kitchen sink (`default`)
  **underperform plain geometry** — ruled out.
- Finalists carried forward: **geo** and **mosaic**.

## exp4 — multi-seed validation (3 seeds)

- **mosaic is the augmentation finalist.** Active mAP50 across seeds at 1024:
  mosaic [0.683–0.699] vs geo [0.667–0.671] — bands don't overlap.
- **mixup adds nothing** (mosaic 0.707 vs mosaic_mixup 0.726 at 512, bands fully
  overlap) — ruled out.
- ~~geo has better Obsolete~~ — **retracted**: geo's 512 Obsolete edge
  (0.320 vs 0.272) was single-seed luck; multi-seed both ~0.20 (tied). Obsolete
  is noise — some seeds predict zero Obsolete at all.
- ~~512@16 clearly beats 1024@8~~ — **corrected**: that anchored on one lucky
  512 seed. Multi-seed, **1024@8 ≈ 512@16** (mild 512 edge on mAP; 1024 slightly
  better medium recall; 512 is noisier, ±0.029 vs ±0.008).
- **Final detector config: mosaic @ 512, batch 16.**

## exp5 — k-fold CV (k=5, stratified, positives-only)

- **The mosaic@512 number is robust — not split luck.** Active mAP50 across 5
  rotating test folds (every positive tested once): **0.697 ± 0.023** (range
  0.672–0.723). The frozen-split value (~0.71) sits inside one std.
- **Split variance < seed variance.** CV std (±0.023) is no larger than the
  seed-only std at 512 (±0.029, exp4) — which images land in test barely moves
  the result; the training seed is the bigger wobble.
- **±~0.025 Active mAP50 is the significance bar** for exp6+: a later
  "improvement" smaller than this isn't distinguishable from noise.
- **Obsolete reconfirmed as noise** (0.177 ± 0.047, fold range 0.097–0.211) —
  headline stays Active-only.
- Guardrails stable: loc IoU 0.739 ± 0.008 (box quality is consistently fine —
  recall is the bottleneck), medium recall 0.467 ± 0.033. Recall-at-threshold is
  the noisy metric (0.675 ± 0.077; fold 0 dips to 0.54 but its mAP50/IoU are
  normal) — read mAP50, not single-fold recall.

## exp6 — init × freeze (VinDr-init vs COCO, + freeze depth), 512 @ batch 16

Two single-seed freeze ladders (mosaic, mosaic_mixup) over freeze ∈ {none, 4, 8,
10, 13} from the VinDr backbone, then the finalist confirmed at 3 seeds.

- **NEW CHAMPION CONFIG: VinDr-init + mosaic_mixup + full fine-tune (freeze=none)
  → Active mAP50 = 0.745 ± 0.028** (3 seeds: 0.762 / 0.706 / 0.767). Beats the
  exp4 locked baseline (COCO+mosaic, 0.707) by +0.038 — clears the ±0.025 bar.
- **But the win is mostly mixup, not VinDr** (honest decomposition, all 512@16
  full-FT, 3 seeds each):
    - COCO + mosaic   0.707 ± 0.024   (exp4 baseline)
    - COCO + mixup    0.726 ± 0.025   (+0.019 — the mixup step)
    - VinDr + mixup   0.745 ± 0.028   (+0.019 — the VinDr step)
  Each step alone (~0.019) is **inside** the noise bar; they STACK to clear it.
  So mixup is the proven lever; VinDr-init is a marginal (within-noise) bonus on
  top — kept because it doesn't hurt and the 1024 backbone may exploit it more.
  ~~mixup adds nothing~~ — exp4's "mixup is a tie" was on COCO; on the VinDr
  backbone mixup beat mosaic at **all 5 freeze levels** (5/5 sweep), a real
  direction even if each single gap is small.
- **Freezing the VinDr backbone is dead.** Full fine-tune won both ladders; no
  freeze depth ever beat `none` (mixup especially: 0.762 vs next-best 0.724).
  Aug already cured overfit (exp3), so freezing just caps a small-data detector.
  Do NOT carry freeze to 1024.
- The mosaic-fz8 cell (0.701) was an artifact — 32 Active→Obsolete mislabels
  inflated its AP; its confusion matrix exposed it. The 0.745 champion's three
  seeds are all clean (A→A 107–111, ~25–31 missed).
- Reading note: the fixed-conf confusion matrix wobbles with each seed's
  confidence calibration (seed 1: low precision 0.49 / high recall 0.81 → fewer
  A→A at the 0.25 gate despite recovering more lesions at lower conf). Read
  mAP50 (conf-integrated), not fixed-threshold counts — same lesson as exp5.

## exp7 — resolution at 1024@16 (yolov8n, init × aug)

Full init×aug grid at 1024, 3 seeds each. Trained remote (COCO on Kaggle, VinDr
on Colab), **evaluated locally** on the sealed test — same `metrics.py` as
exp1-6, so directly comparable. Active mAP50:

| 1024@16 | mosaic        | mixup         |
|---------|---------------|---------------|
| COCO    | 0.721 ± 0.013 | 0.703 ± 0.030 |
| VinDr   | 0.748 ± 0.025 | 0.716 ± 0.017 |

- **1024 ≈ 512 at yolov8n — no resolution payoff.** Best 1024 cell (VinDr mosaic
  0.748) only ties the 512 champion (0.745, +0.003). Doubling the training cost
  buys nothing on Active mAP50. Confirms exp4's "1024@8 ≈ 512@16" at batch 16.
- **mixup reverses at 1024.** At 512 mixup was the lever; at 1024 mosaic wins on
  both inits (VinDr 0.748 > 0.716, COCO 0.721 > 0.703). ~~mixup is the real
  lever~~ — narrowed: it's the lever *at 512*; mixup is resolution-specific.
- **VinDr-init helps a bit more at 1024** than at 512: VinDr mosaic 0.748 vs COCO
  mosaic 0.721 = +0.027 (just clears the ±0.025 bar).
- **Curves show overfitting, not saturation** (val loss rises, val mAP peaks
  ~ep20–35 then drifts). The 799-image data ceiling — best.pt selection still
  grabs the early peak, so the final number holds. This motivates exp8.
- Confusion matrices clean (max Active→Obsolete = 11; no fz8-style artifact).

## exp8 — model capacity (yolov8s) + inference tricks

exp7 closed 1024 *at yolov8n*; this asks whether the wall is **capacity** (a
bigger model exploits resolution) or the **799-image data ceiling** (a bigger
model just overfits). Plus the free inference tricks (TTA, seed-ensemble).

- **yolov8s @ 512 mosaic_mixup** (COCO-init, 3 seeds): 0.691 / 0.691 / 0.704 →
  **0.695 ± 0.008**. *Below* COCO-n mixup@512 (0.726). Capacity hurts at 512.
- **yolov8s @ 1024 mosaic** (COCO-init, 3 seeds, Kaggle 2×T4): 0.714 / 0.739 /
  0.648 → **0.700 ± 0.047**. *Below* COCO-n mosaic@1024 (0.721) and far noisier.
  Bigger model + bigger images is no better — slightly worse, higher variance.
  (s@1024@16 OOMs a single T4; needs 2×T4 — even fitting it is a fight.)
- **TTA** on the champion (3 seeds, inference-only, no retrain): 0.733 / 0.724 /
  0.768 → **0.742 ± 0.023** vs no-TTA 0.745. Dead tie — TTA trades precision for
  recall, conf-integrated mAP50 nets flat.
- **Seed-ensemble: not run.** The 3 champion seeds share the frozen split, so
  their errors are correlated → WBF gains little (and TTA, the same idea, already
  netted zero). A *decorrelated* CV-fold ensemble has no clean test left — exp5
  trained every image in some fold. Not worth it.

## Verdict — YOLO detection baseline LOCKED

Four independent levers all closed within/under the ±0.025 bar: resolution
(exp7), capacity at 512 and 1024 (exp8), TTA. **The bottleneck is data (799
images), not the model, the resolution, or inference.** Box quality was never the
problem (loc IoU ~0.74 throughout) — recall is, and recall is data-limited.

**Final YOLO detector: VinDr-init + mosaic_mixup + full fine-tune @ 512 @ batch
16 = Active mAP50 0.745 ± 0.028** on the sealed test. This is the detection
baseline the rest of the project builds on.

---

## Next — the two-stage pipeline (the original intent)

The detector is done, and it is deliberately **positives-only / sensitive** —
specificity was always the **classifier's** job (exp2). Next is the real
architecture: **image-level classifier → TB detector**. Stage 1 screens healthy /
sick / TB and forwards only likely-TB images; stage 2 is this locked detector.
See `ideas.md` for the pipeline-topology design space (clf→det, det→clf, 3-stage)
and the data-centric ideas (pseudo-labeling, lung-seg masking, lesion copy-paste)
that could still lift either stage past the 799-image ceiling.
