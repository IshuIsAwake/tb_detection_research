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

## exp8 — model capacity n → s → m (IN PROGRESS)

exp7 closed 1024 *at yolov8n only*. Open question: is the wall **capacity** (then
yolov8s/m would exploit 1024) or the **799-image data ceiling** (then a bigger
model just overfits harder — which the exp7 curves hint at)?

- Step 1 (running locally): COCO-init yolov8s @ 512 mixup ×3, judged against
  COCO-**n** mixup@512 (0.726) to isolate capacity. Batch 16 fits 6 GB (smoke-
  checked).
- Step 2: yolov8s @ 1024 on Colab (mosaic + mixup ×3). A VinDr-s backbone is
  gated on whether yolov8s shows any gain first.
- Judge: Active mAP50 on the sealed test, ±0.025 bar. Champion to beat = 0.745.

---

## Next / later

- If capacity helps: pretrain a yolov8s VinDr backbone, re-run the best cell.
- If not: TTA + seed-ensembling as final polish.
- Infra unchanged: train remote, **eval local** — `kaggle/tbx_train.py` (+ the
  Colab cell) emit best.pt, `yolo_experiments/eval_weights.py` scores it on the
  sealed test. See `ideas.md` for parked directions (pipeline topology, lung
  segmentation, custom lesion copy-paste, texture enhancement).
