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

---

## Next

- **exp5 — k-fold CV**: test-set-luck / split variance on the final config.
- **VinDr-pretrained weights**: supervised YOLO pretrain on VinDr → init for
  TBX11K (Kaggle, parallel). **Freezing** ablation pairs with this.
