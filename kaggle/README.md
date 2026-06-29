# VinDr pretrain (Kaggle)

Supervised `yolov8n` pretrain on VinDr-CXR to get a domain-matched backbone for
the TBX11K detector. Runs on Kaggle's free 2×T4; the local 6 GB card can't.

The output we keep is **`best.pt`**, not VinDr metrics. VinDr is a hard detection
benchmark — a nano model at 512 lands ~0.10–0.20 mAP50 and that's fine. The real
test is exp6 (VinDr-init vs COCO-init on TBX11K Active mAP50).

The same script makes **both** backbones — set `IMGSZ` and attach the matching
dataset. The run name carries the resolution, so they never collide:

| resolution | dataset | `IMGSZ` | `BATCH` (2×T4) | feeds |
|---|---|---|---|---|
| 512 (default) | [awsaf49/vinbigdata-512-image-dataset](https://www.kaggle.com/datasets/awsaf49/vinbigdata-512-image-dataset) | 512 | 64 | exp6 (512 TBX11K) — **done** |
| 1024 | [awsaf49/vinbigdata-1024-image-dataset](https://www.kaggle.com/datasets/awsaf49/vinbigdata-1024-image-dataset) | 1024 | 16 | exp7/8 (1024@16 TBX11K, Kaggle) |

At 1024 the images are 4× the pixels, so drop `BATCH` from 64 → **16** (8/GPU on
2×T4); push to 24/32 only if it fits. Everything else is identical — no code
change between the two.

## What to attach

Recommended: a VinDr dataset from the table above. Any square VinDr set works if
it has:
- images named `<image_id>.png`/`.jpg`
- a CSV with `image_id, class_id, x_min, y_min, x_max, y_max` (`class_id 14` =
  "No finding"). If it also has `width`/`height` columns (awsaf49 does), the
  converter normalises by those (original coords); otherwise by image pixel size.
  A square→square resize preserves normalised coords, so 512 and 1024 convert
  identically — the **overlay gate HALTs** if a dataset's coords don't fit.

The script auto-discovers the CSV and images under `/kaggle/input` — no slug is
hardcoded. If the coords don't fit the images, the **overlay gate halts** before
wasting a run.

## Steps

1. New Kaggle Notebook → Accelerator **GPU T4 ×2**, Internet **ON** (first run
   only, so Ultralytics can fetch `yolov8n.pt`).
2. Add the VinDr 512 dataset (above) as input.
3. First cell: `!pip -q install ultralytics`
4. Paste `vindr_pretrain.py` into a cell (or upload it and `%run vindr_pretrain.py`).
5. **Smoke first** to prove plumbing (at the resolution you're running):
   ```python
   import os; os.environ["SMOKE"] = "1"; os.environ["IMGSZ"] = "1024"  # or 512
   %run vindr_pretrain.py
   ```
6. **Eyeball `/kaggle/working/overlay_check/*.png`** — boxes must sit on
   anatomy. This is the one check that matters. If boxes are off/garbage, the
   CSV↔image coordinate pairing is wrong — fix the dataset, don't train.
7. Full run: clear `SMOKE`, set `IMGSZ`/`BATCH` per the table, `%run
   vindr_pretrain.py`. It early-stops on VinDr val (`patience=20`), so it ends
   when features converge, not at a hard cap.
8. Download `vindr_runs/vindr_yolov8n_mosaic<IMGSZ>/weights/best.pt`
   (`...mosaic512` or `...mosaic1024`).

## Knobs (env vars)

`IMGSZ=512 EPOCHS=80 BATCH=64 PATIENCE=20 DEVICE=0,1 VAL_FRAC=0.08 SEED=0`.
For 1024: `IMGSZ=1024 BATCH=16` (see the table). If multi-GPU DDP errors in the
notebook, set `DEVICE=0` (single T4, slower).

Locked, do not change: `MODEL=yolov8n.pt` (must match downstream or weights
won't load) and the `mosaic` aug (copied from `yolo_common/aug.py`).

## Bringing it back

`best.pt` is gitignored (`*.pt`). Save it under `weights/pretrain/` in the repo,
keeping the resolution in the name so the two backbones don't clash:
- 512 → `weights/pretrain/vindr_supervised_best.pt`      (the exp6 init — already here)
- 1024 → `weights/pretrain/vindr_supervised_1024_best.pt`  (for exp7/8)

Then a TBX11K run uses it with **zero harness change** via `MODEL=` — Ultralytics
loads the VinDr backbone+neck and reinitialises the 14→2 class head automatically:

```bash
# 1024 final (exp7/8) — runs on Kaggle, local 6GB can't do 1024@16
MODEL=weights/pretrain/vindr_supervised_1024_best.pt AUG_LEVEL=mosaic \
  BATCH=16 EPOCHS=200 python yolo_experiments/exp6_init_freeze.py \
  --aug-level mosaic --freeze <winning depth> --imgsz 1024 --batch 16
```

The winning freeze depth comes from the exp6 512 screen; the 1024 backbone just
swaps in as the init.

---

# TBX11K detector training on Colab (`tbx_train.py` + `colab_cell.py`)

For the **1024@16 VinDr-init** batch (mosaic ×3 + mosaic_mixup ×3 = 6 runs) we use
**Colab, not Kaggle**: it's a morning job, free Colab gives **one GPU runtime per
account**, and 3 accounts run 3 seeds in parallel. Each account runs **both augs
for its seed** (2 jobs back-to-back, ~5 h). Kaggle stays free for overnight runs.

Free Colab has **no background execution** — keep the tab open. `tbx_train.py`
checkpoints to Drive every epoch (`PROJECT_ROOT=Drive`), so a disconnect costs
only the in-flight epoch: re-run the same cell and it **resumes from `last.pt`**
(or **skips** a job that wrote its `DONE` sentinel).

## Prep the Drive folder (once, locally)

```bash
# assemble the bundle (same three files Kaggle needs)
mkdir -p upload && cp yolo_datasets/splits.json upload/
cp -r data/TBX11K/imgs/tb upload/tb
cp -r yolo_datasets/labels_all upload/labels_all
```

Then upload to one Drive folder (default `MyDrive/tb/`):
- everything in `upload/` → `MyDrive/tb/` (so `splits.json`, `tb/`, `labels_all/`)
- the 1024 VinDr backbone → `MyDrive/tb/vindr_pretrain_1024.pt`
- `kaggle/tbx_train.py` → `MyDrive/tb/tbx_train.py`

> Note: the file is `vindr_pretrain_1024.pt` (what's in `weights/pretrain/`), not
> the `vindr_supervised_1024_best.pt` name the VinDr README predicted — the cell
> uses the real filename.

## Run (each account)

New Colab notebook → Runtime → **T4 GPU**. Paste `kaggle/colab_cell.py` into a
cell, set `SEED` (A=0, B=1, C=2), run. Smoke first if you want plumbing proof:
add `EPOCHS="2"` to the `env` dict, confirm one job completes, then clear it.

The 6 best.pt land at `MyDrive/tb/colab_runs/tbx_vindr1024_<aug>_1024_b16_s<seed>/weights/best.pt`.

## Bringing it back — eval LOCAL

Download the 6 `best.pt`, then score on the sealed test (same as exp1-6):

```bash
for aug in mosaic mosaic_mixup; do for s in 0 1 2; do
  python yolo_experiments/eval_weights.py \
    --weights <best.pt for $aug s$s> --imgsz 1024 \
    --name "tbx_vindr1024_${aug}_1024_b16_s${s}" \
    --meta "{\"model\":\"yolov8n\",\"imgsz\":1024,\"batch\":16,\"augmentation\":\"${aug}\",\"seed\":${s},\"init\":\"vindr1024\",\"trained\":\"colab\"}"
done; done
```

