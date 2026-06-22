# RESULTS.md — chronological experiment log

The honest record of what we ran and what we concluded. **Retractions are
struck through, not deleted** — the record of *why* a conclusion was wrong is
the asset. Where this file and code disagree, this file is the authority on what
was actually measured.

Headline numbers per run live in `yolo_experiments/results/<run>/metrics.json`;
this file is the narrative.

---

## exp1 — YOLOv8n detection floor (diagnostic) · status: DONE (512@16, 512@8, 1024@8)

`yolo_experiments/exp1_yolo_baseline.py`. Unoptimised COCO-pretrained YOLOv8n,
positives-only training, augmentation OFF, full fine-tune (no freeze), AMP OFF,
deterministic. RTX 3050 6GB, 100 epochs. Same sealed 360 blackbox throughout.

Frozen split (seed 0): 559 train / 119 val positives; sealed blackbox = 121 TB
+ 120 sick + 120 healthy. 240 negative ids reserved (see leakage rule).

### Three runs (1024 caps at batch 8 on 6GB; 512 also run at batch 8 to de-confound)

| run | batch | mAP50 | mAP50-95 | recall | Active mAP | Obsolete mAP | loc IoU | sickFA@.25 | healthyFA@.25 |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 16 | 0.370 | 0.164 | 0.443 | 0.607 | 0.133 | 0.741 | 0.533 | 0.267 |
| 512 | 8 | 0.318 | 0.139 | 0.389 | 0.541 | 0.095 | 0.752 | 0.342 | 0.083 |
| 1024 | 8 | 0.357 | 0.159 | **0.498** | 0.591 | 0.122 | 0.728 | **0.233** | 0.100 |

Recall by GT lesion size (≈same across runs): small <1% **0.00** (n=2) · medium
1-5% **~0.32** (n=72) · large >5% **~0.64** (n=105).

### Diagnostic read

1. **Active ≫ Obsolete** at all settings (mAP ~0.59 vs ~0.12). Obsolete barely
   learned: 178 train boxes / 35 test instances vs Active's 724/146. Class
   imbalance, not localisation. Resolution did **not** fix it.
2. **Recall is the bottleneck, not box quality.** On matched pairs IoU≈0.73-0.75,
   IoG≈0.83-0.89 (good); but only ~90 of 179 GT matched at conf 0.25. The job is
   finding lesions, not tightening boxes.
3. **Confuses sick (non-TB pathology) with TB**, ~2-3× the healthy false-alarm
   rate — the signal the harness was built for. The model fires on abnormal lungs
   regardless of TB. 1024 narrows it (more specific at matched sensitivity).
4. **Strong size dependence** (large ≫ medium ≫ small≈0). Resolution did not move
   it much; motivates tiling / class-balance / more data, not just pixels.

### Resolution & batch findings (de-confounded)

- **At equal batch (8), 1024 > 512 on every detection metric** (mAP50, mAP50-95,
  recall, precision, both per-class). 512 wins only box-tightness-on-hits (within
  noise; irrelevant since recall is the bottleneck). The earlier "512 looks best"
  was the **batch-16 advantage of that run, not the resolution.**
- **Batch is a real lever and shifts the operating point.** 512@16 vs 512@8: more
  aggressive detector — higher recall AND higher false alarms. So batch-16's higher
  mAP is partly "mAP rewards recall," not strictly better. Compare the
  sensitivity/specificity tradeoff, never a single-conf row.
- **Overfits early.** Live 1024 loss trace: train loss kept dropping to ep100, but
  **val cls-loss bottomed ~ep12 then nearly doubled, and val mAP50 peaked at
  epoch 12** then declined. `best.pt` (peak fitness) saves us, but the floor's
  real problem is overfitting, not undertraining — more epochs HURT. Levers:
  augmentation (biggest), more data, partial freeze.

### Decision for exp2+

Iterate ideas at **512 batch 16** (fast workhorse, ~11 min); **validate finalists
at 1024 batch 8** (quality ceiling on this GPU). Resolution is settled as a
ladder, not a single choice.

---

## exp2 — negatives in detector training · status: DONE (ratio sweep 0.25 / 0.5 / 1.0)

`yolo_experiments/exp2_negatives.py`. Exactly the exp1 floor (yolov8n, 512,
batch 16, aug OFF, no freeze) with ONE change: background (no-TB) images added
to TRAIN, drawn fixed-seed from the non-reserved pool (240 blackbox negatives
excluded; leakage-guarded). val stays positives-only. Same sealed 360 blackbox.

Compared against the batch-matched **512 @ batch 16 positives-only** baseline:

| run | neg:pos | mAP50 | recall | TB@.10 | sickFA@.10 | healthyFA@.10 | TB@.25 | sickFA@.25 |
|---|---|---|---|---|---|---|---|---|
| pos-only | 0 | **0.370** | **0.443** | **0.967** | 0.783 | 0.542 | 0.868 | 0.533 |
| exp2 | 0.25 | 0.299 | 0.383 | 0.488 | 0.058 | 0.008 | 0.405 | 0.033 |
| exp2 | 0.5 | 0.314 | 0.338 | 0.603 | 0.075 | 0.017 | 0.496 | 0.042 |
| exp2 | 1.0 | 0.269 | 0.263 | 0.620 | 0.025 | 0.000 | 0.545 | 0.008 |

### Findings

1. **False alarms collapse, and the benefit saturates almost instantly.** Even
   the smallest dose (0.25:1, ~20% background) drops sick-FA 0.783→0.058 and
   healthy-FA →~0. Going to 1.0 barely improves it further. The healthy-vs-sick
   gap (the exp1 signal) essentially vanishes.
2. **But negatives impose a hard sensitivity ceiling of ~60%.** Positives-only
   reaches 97% TB detection at low conf; every negatives run caps at ~49–62% TB
   detection **at any threshold** — the model stops firing on a large fraction of
   TB images. Not a dial-back-able threshold shift; the sensitivity is gone.
3. **The ratio (0.25 / 0.5 / 1.0) is unresolvable from this sweep.** Detection
   metrics are non-monotonic across ratios (mAP 0.299/0.314/0.269; TB@.10
   0.488/0.603/0.620) — single-seed run-to-run noise (no aug, overfit by ep12)
   is as large as the ratio effect. Can't honestly pick a "best" ratio here.
4. Box quality on hits unchanged (IoU ~0.74) — this is a firing-behaviour change,
   not a localisation change.

### Verdict — negatives belong in the CLASSIFIER, not the detector

The trade is "specificity for sensitivity," and that is the wrong trade for the
detector in the two-stage **classifier → detector** plan: the classifier already
gates out healthy/sick, so the detector should stay **sensitive** (find every
lesion on flagged images). Capping detector sensitivity at ~60% to buy
specificity the classifier will provide is solving the problem in the wrong
stage. **Keep the detector positives-only/sensitive; let the classifier own
specificity.** Negatives lever closed for the detector.

The detector's real bottleneck remains **recall** (small/medium lesions,
Obsolete class) plus the **overfitting** (peaks ~ep12) that is also the source of
the cross-run noise above. Biggest untouched lever for both: **augmentation**.

---

## Next

- **exp3 — augmentation** (positives-only, 512@16): attacks recall AND the
  overfitting/noise. Design to be discussed before implementing.
- **SSL pretraining (DINO / BYOL)** on Kaggle 2×T4, in parallel — get pretrained
  CXR weights ready while local work continues (local 6GB can't train it).
