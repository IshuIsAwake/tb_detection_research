# Claude Code Task — `exp1_yolo_baseline.py`

## Project context (read first)

We're building TB detection on chest X-rays. The long-term plan is a two-stage
pipeline (image-level classifier → TB detector); right now we are building the
**detection baseline only**. A dataset research report, `data_research_report.md`,
is in the project folder — read it before doing anything; it has the authoritative
dataset facts.

This file is **exp1**: an unoptimised, ImageNet/COCO-pretrained **YOLOv8n** detector.
Its purpose is **diagnostic, not performance**. We deliberately do not optimise it.
We want to see exactly where a naive baseline fails — small-lesion misses,
Active/Obsolete confusion, and false alarms on abnormal-but-not-TB lungs — so that
later experiments know precisely what to fix. Do not "help" it with augmentation,
tuning, or cleverness.

## What to build

A **single self-contained file**: `yolo_baseline/exp1_yolo_baseline.py`.

Philosophy (mirror the house style described below): **one file = one ablation**.
The file is run twice — `--imgsz 512` produces **exp1a**, `--imgsz 1024` produces
**exp1b** — and the *only* variable between them is input resolution. There is no
separate "1c": both runs evaluate on the same sealed 360-image test set, so we read
the full picture for each resolution directly and compare.

## House style to follow

- The **module docstring is the experiment spec** — state what it does, why, the exact
  run commands, and how to read the results table. Someone should understand the whole
  experiment from the docstring alone.
- **All knobs env-overridable** with sane defaults; `imgsz` also via CLI.
  Defaults: `EPOCHS=100`, `BATCH=16`, `SEED=0`, `CONF_THRESHOLDS=0.10,0.25,0.50`.
- A **smoke mode** (`TB_SMOKE=1`) that runs 1–2 epochs on a tiny subset to validate
  the full plumbing end-to-end without a real train.
- **Fully deterministic**: seed Python, NumPy, torch, and the ultralytics seed.
- **No augmentation, no cross-validation, AMP off.** This is a deliberate floor.
- **Structured results** per run: `metrics.json` (machine-readable) + `summary.txt`
  (human table) + the ultralytics run artifacts.
- Keep it one readable, sectioned file. Implement the custom metrics inline. You may
  `pip install ultralytics` if it's missing (PyPI is reachable).

## Step 0 — Inventory the data, halt on contradiction

The TBX11K data is already in YOLO format in the folder, but **discover the exact
layout — do not hardcode paths**. Produce and print an inventory:

- Folder names and per-folder image counts (expect roughly: Healthy ~3,800,
  Sick-non-TB ~3,800, TB ~800, Test ~3,302, External ~1,106).
- Count of non-empty label files (the TB positives — expect ~800).
- Box-level class-id histogram. **Verify** there are exactly 2 box classes and that
  the mapping is `0 = ActiveTuberculosis`, `1 = ObsoletePulmonaryTuberculosis`
  (expect ~972 Active boxes, ~239 Obsolete, ~1,211 total). Verify — do not assume.

If anything contradicts `data_research_report.md` (flipped class mapping, wrong
counts, missing folders), **STOP and print the discrepancy.** Do not guess or
silently "fix" it.

Note: the External folder (DA, DB, Montgomery, Shenzhen) is classification-only,
no boxes, and is **out of scope for exp1** — ignore it here. The official Test
folder has no labels (withheld) — also ignored; our test set is built ourselves below.

## Step 1 — Build the canonical split once, persist it, never regenerate

Write `yolo_baseline/_datasets/splits.json` with a **build-or-load** pattern: if it
exists, load it; never rebuild. This split is the frozen reference for **every future
experiment**, so getting it stable now matters.

- **Positives** = the ~800 TB images (non-empty labels). This is the only training data.
- Stratify by `{Active-only, Obsolete-only, both}`, fixed seed, **70/15/15** →
  `train` / `val` / `test_positives` (≈ 560 / 120 / 120).
- **Sealed test set** (the permanent blackbox) = `test_positives` (120 TB)
  + 120 Sick-non-TB + 120 Healthy, the negatives sampled fixed-seed from their folders.
- Record in `splits.json`: `train_ids`, `val_ids`, `test_positive_ids`, and
  `blackbox_negative_ids` (the 120 sick + 120 healthy, by filename/ID).

**CRITICAL leakage rule, document it in the file and in `splits.json`:** the
`blackbox_negative_ids` are **reserved** — every future experiment that adds negatives
to *training* must draw from the remaining ~3,680 sick and ~3,680 healthy and must
exclude these 240. Otherwise a future experiment trains on images that sit in the
final test set and silently breaks the blackbox.

Build `data.yaml` / image-list files from the split. Train on `train`; ultralytics
`val` = `val_positives` (monitoring/best-weight selection only). The diagnostic eval
(Step 3) runs on the full 360 sealed set. **Same images for both resolutions** — do
not make 512/1024 copies; `imgsz` handles it and ultralytics resizes internally.

## Step 2 — Train (per `imgsz`)

- `yolov8n.pt`, `imgsz` from CLI, `epochs`/`batch`/`seed` from env.
- Train on `train` positives; ultralytics `val` = `val_positives`.
- **Augmentation fully OFF** — explicitly zero/disable every ultralytics aug
  (`mosaic=0, mixup=0, copy_paste=0, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0,
  translate=0, scale=0, shear=0, perspective=0, fliplr=0, flipud=0, erasing=0`).
  ultralytics augments by default; this must be deliberately turned off.
- `amp=False`, deterministic seeding.

## Step 3 — Evaluate on the sealed 360 (the diagnostic table)

Do **not** rely on ultralytics' built-in val alone — build a custom evaluation over
the full 360 and assemble one table. Include:

**Detection (threshold-free AP):** `mAP@.5`, `mAP@.5:.95`, precision, recall —
overall and per class (Active, Obsolete).

**Localization quality** (predictions at `conf≥0.25`, greedily matched to GT, keep
matches with `IoU≥0.5`), mean over matched pairs, overall and per class:
- `IoU  = |P∩G| / |P∪G|`
- `IoP  = |P∩G| / |P|`  (over-prediction: low ⇒ boxes too big)
- `IoG  = |P∩G| / |G|`  (coverage: low ⇒ boxes too small / partial)
- also report `n_matched`.

**Screening / false alarm**, computed at each of `conf ∈ {0.10, 0.25, 0.50}`
(an image is "flagged" if it has ≥1 prediction at that threshold):
- TB detection rate = flagged / 120 TB
- Healthy false-alarm rate = flagged / 120 Healthy
- Sick-non-TB false-alarm rate = flagged / 120 Sick
- avg predictions per image, per group (over-firing magnitude)
- print raw counts too: `"X of 120 TB flagged; M of 120 healthy; K of 120 sick"`.

**By lesion size** (the suspected failure): bucket GT boxes by area
(small <1% img area, medium 1–5%, large >5% — pick sensible cuts and state them),
and report **recall (detection rate) per bucket** with the GT count per bucket.
Surface small-lesion recall prominently.

Add a one-line note in `summary.txt`: a positives-only model is **expected** to
over-fire on negatives, so high false-alarm is the diagnostic, not a bug — the
**healthy-vs-sick false-alarm gap** is the key signal (a larger gap means the model
is confusing other pathology with TB).

## Step 4 — Write results (built for the GitHub / Antigravity page)

Per run, write to `yolo_baseline/results/exp1a_512/` or `.../exp1b_1024/` (folder named
by `imgsz`):

- `metrics.json` — stable schema (below), so the webpage renders all experiments
  uniformly.
- `summary.txt` — the human-readable table.
- the ultralytics run dir (`best.pt`, PR/F1 curves, confusion matrix).
- a config snapshot.

`metrics.json` schema (stable keys):

```json
{
  "experiment": "exp1a_512",
  "config": {"model":"yolov8n.pt","imgsz":512,"epochs":100,"batch":16,"seed":0,
             "augmentation":false,"amp":false,"conf_thresholds":[0.10,0.25,0.50]},
  "dataset": {"splits_file":"...","split_seed":0,
              "train_positives":560,"val_positives":120,
              "blackbox":{"tb":120,"sick_non_tb":120,"healthy":120},
              "box_counts":{"active":972,"obsolete":239},
              "class_map":{"0":"ActiveTuberculosis","1":"ObsoletePulmonaryTuberculosis"}},
  "environment": {"ultralytics":"...","torch":"...","gpu":"...","timestamp":"...","git_commit":"..."},
  "metrics": {
    "detection":   {"overall":{"mAP50":0,"mAP50_95":0,"precision":0,"recall":0},
                    "active":{...}, "obsolete":{...}},
    "localization":{"overall":{"iou":0,"iop":0,"iog":0,"n_matched":0},
                    "active":{...}, "obsolete":{...}},
    "screening":   {"0.25":{"tb_detect_rate":0,"tb_flagged":"X/120",
                            "healthy_false_alarm":0,"healthy_flagged":"M/120",
                            "sick_false_alarm":0,"sick_flagged":"K/120",
                            "avg_preds_per_img":{"tb":0,"healthy":0,"sick":0}},
                    "0.10":{...}, "0.50":{...}},
    "by_size":     {"small":{"recall":0,"n":0},"medium":{"recall":0,"n":0},"large":{"recall":0,"n":0}}
  },
  "timing": {"train_sec":0,"eval_sec":0},
  "artifacts": {"weights":"...","pr_curve":"...","confusion_matrix":"..."}
}
```

## Invocation

```
python yolo_baseline/exp1_yolo_baseline.py --imgsz 512     # exp1a
python yolo_baseline/exp1_yolo_baseline.py --imgsz 1024    # exp1b
TB_SMOKE=1 python yolo_baseline/exp1_yolo_baseline.py --imgsz 512   # plumbing test
```

## Guardrails (do not skip)

- Read `data_research_report.md` and run the Step 0 inventory **before** writing the
  training code; halt on any contradiction rather than guessing.
- Do **not** add augmentation, cross-validation, or any tuning — this is a deliberate
  unoptimised floor.
- Build `splits.json` once and freeze it; reserve `blackbox_negative_ids`.
- Deterministic seeding throughout.
- One self-contained file is the deliverable (`splits.json`, `data.yaml`, and result
  folders are produced by running it — that's expected).
