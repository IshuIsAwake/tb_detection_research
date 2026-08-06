# Kaggle launch — K1, K2, K3 (2026-08-06)

Three sessions, launched while the local 3050 runs the 15 CV folds. K4
(byolvindr + kornia) is being built in a separate chat.

| session | run | GPU time |
|---|---|---|
| K1 | image-level classifier: `byolvindr`, then `imagenet` control | ~2 × 1.5 h |
| K2 | detector `cocovindr` + mosaic_mixup + **w500** | ~1 h |
| K3 | detector `byolvindr` + mosaic_mixup + **w500** | ~1 h |

K2 and K3 fill the two `sc_` mosaic_mixup cells that died on the NaN guard —
their logs exist (`sc_cocovindr_mosaic_mixup_s0.log`,
`sc_byolvindr_mosaic_mixup_s0.log`) but no result directories, because they
crashed. K3 is also the **control** for K4's kornia cell: same init, same aug,
the only difference is kornia. Keep every other knob identical between them.

---

## ⚠⚠ The three things that silently ruin a Kaggle run

**1. `splits.json` must be UPLOADED, never rebuilt.**
`_stratified_split` seeds its shuffle with `hash(sig)`, and Python randomises
string hashing per process — `hash('active') % 997` measured 186 / 201 / 516 on
three consecutive runs of the same machine. A rebuild produces a **different
sealed TB test set**, so TB images sealed locally become training data on Kaggle.
`TB_REQUIRE_SPLITS=1` (below) turns that from silent to a halt. Set it every time.

**2. `/kaggle/working` does not persist.** Download artifacts before the session
ends or the run is gone. See the download cell at the bottom.

**3. Config typed into a cell is unrecoverable.** Every Kaggle breakage in this
project has the same shape: a treatment/control pair that silently disagreed on
one knob, with no record of the launch (see RESULTS.md § "The Kaggle failure
mode" — `BN_RECALIB` differed between two rungs and it took a day and a
`num_batches_tracked` count to find). The cells below `echo` the resolved config
into the log so the launch is recoverable from the downloaded artifacts.

---

## Payload — upload once, attach to all three sessions

Create these as Kaggle datasets (private):

| dataset | contents | sha256 |
|---|---|---|
| `tb-splits` | `yolo_datasets/splits.json` | `2bd2ac7a7aa71566…4fa54220` |
| `tb-weights` | `weights/pretrain/vindr_cocoinit_matched_s0.pt` | `385497a811df8dfa…308f431d` |
| | `weights/pretrain/vindr_byolinit_matched_s0.pt` | `a944c7928da8df1e…22d2dee3` |
| `tb-code` | the repo — **K2/K3 only**; `retinanet.py` needs `yolo_common/`. K1's `tbx_classifier.py` is self-contained, so `%%writefile` it and skip this. | — |
| `tbx11k` | the TBX11K dataset (already uploaded from earlier runs) | — |

Full checksums: `sha256sum yolo_datasets/splits.json weights/pretrain/vindr_*.pt`

---

## Cell 1 — common setup (identical in all three sessions)

```python
import os, subprocess, hashlib, pathlib, shutil

REPO = "/kaggle/working/tb"
shutil.copytree("/kaggle/input/tb-code", REPO, dirs_exist_ok=True)

# GEN_ROOT must be WRITABLE (labels are generated into it) but must already
# contain the frozen splits.json — /kaggle/input is read-only, so copy it out.
GEN = pathlib.Path("/kaggle/working/yolo_datasets"); GEN.mkdir(parents=True, exist_ok=True)
shutil.copy("/kaggle/input/tb-splits/splits.json", GEN / "splits.json")

got = hashlib.sha256((GEN / "splits.json").read_bytes()).hexdigest()
EXPECT = "2bd2ac7a7aa715666d7e6e4814244b95c9f3188c23604af3b47227274fa54220"
assert got == EXPECT, f"splits.json MISMATCH\n got {got}\n exp {EXPECT}"
print("splits.json verified ✓")

os.environ.update({
    "TB_DATA_ROOT":     "/kaggle/input/tbx11k/TBX11K",
    "TB_GEN_ROOT":      str(GEN),
    "TB_RESULTS_ROOT":  "/kaggle/working/results",
    "TB_REQUIRE_SPLITS": "1",      # halt rather than rebuild a different split
})
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Expected: `splits.json verified ✓` and a GPU line. If the assert fires, the
dataset is stale — re-upload, do not proceed.

---

## Cell 2 — K1: the classifier

```python
%cd /kaggle/working/tb
!INIT=byolvindr EPOCHS=30 BATCH=32 python kaggle/tbx_classifier.py \
    --name cls_byolvindr 2>&1 | tee /kaggle/working/cls_byolvindr.log
```

Then, in the same session once it finishes (the control — do not skip it, it is
the only thing that makes the SSL chain's value reportable):

```python
!INIT=imagenet EPOCHS=30 BATCH=32 python kaggle/tbx_classifier.py \
    --name cls_imagenet_control 2>&1 | tee /kaggle/working/cls_imagenet.log
```

**Check in the first 60 seconds:**
- `[leak-check] no reserved blackbox negative in train/val ✓` — if this raises
  `SystemExit` instead, the 240 sealed negatives reached training. Stop.
- `[init] byolvindr: 318 trunk tensors loaded, 0 unexpected, 0 missing` — fewer
  than 300 halts by design. `imagenet` prints `(CONTROL arm)` instead.
- `train n=…` should be ≈7,400 with all four classes present.
- `val AUROC(tb|sick)` should climb off ~0.5 within a few epochs.

⚠ Selection is on **AUROC(tb-vs-sick)**, not accuracy — TB-vs-other-pathology is
the real task; TB-vs-healthy is easy. Read the vs-sick number first.

---

## Cell 2 — K2: detector, cocovindr + mosaic_mixup

```python
%cd /kaggle/working/tb
!python yolo_experiments/retinanet.py --single-class \
    --batch 8 --accum 2 --aug mosaic_mixup \
    --freeze-bn --bn-recalib 50 --warmup-steps 500 \
    --backbone-weights /kaggle/input/tb-weights/vindr_cocoinit_matched_s0.pt \
    --name retinanet_sc_cocovindr_mosaicmix_w500_s0 \
    2>&1 | tee /kaggle/working/sc_cocovindr_mosaicmix_w500_s0.log
```

## Cell 2 — K3: detector, byolvindr + mosaic_mixup (K4's control)

```python
%cd /kaggle/working/tb
!python yolo_experiments/retinanet.py --single-class \
    --batch 8 --accum 2 --aug mosaic_mixup \
    --freeze-bn --bn-recalib 50 --warmup-steps 500 \
    --backbone-weights /kaggle/input/tb-weights/vindr_byolinit_matched_s0.pt \
    --name retinanet_sc_byolvindr_mosaicmix_w500_s0 \
    2>&1 | tee /kaggle/working/sc_byolvindr_mosaicmix_w500_s0.log
```

**Check in the first 2 minutes (both K2 and K3):**
- `[sched] lr=0.01 eff_batch=16 warmup=500 optimizer steps (~14.3 epochs)` —
  **if it says `warmup=34 … [legacy]` the flag did not take. Kill it.** Without
  w500 these exact cells crashed 6 of 8 times.
- `[backbone] … 334 tensors loaded, 0 missing, 0 unexpected` — a silent fallback
  to plain COCO would look like a real run and score like a mediocre one.
- `[bn] recalibrated 53 … [bn] froze 53` — both lines, in that order.
- `train_positives=559 val_positives=119 sealed_eval_items=361` — **not** a
  `[kfold]` block. K2/K3 are frozen-split runs, not folds. (361 = 121 tb + 120
  sick + 120 healthy; verified against `sc_cocovindr_s0.log`.)
- Epochs 2–3 are where every NaN crash landed. Past epoch 5 it is safe to leave.

⚠ Do not change `--batch 8 --accum 2`. A bigger Kaggle GPU tempts you to raise
it, but eff_batch 16 is what every recorded cell used, and K3 must match K4
exactly for the kornia contrast to be one-knob.

---

## Cell 3 — download before the session dies

```python
import shutil
shutil.make_archive("/kaggle/working/artifacts", "zip", "/kaggle/working/results")
print("download /kaggle/working/artifacts.zip AND the .log files")
```

Then locally, unpack each run into `yolo_experiments/results/<name>/` so
`metrics.json` sits where the collectors expect it.

⚠ `best.pt` is ~146 MB per detector run. If the zip is too big to download,
grab `metrics.json` + `summary.txt` + `history.jsonl` first — those are the
result. The weights only matter if you plan to run inference again.

---

## What each session produces

- **K1** → `cls_byolvindr` and `cls_imagenet_control` `metrics.json`, each with
  the sealed-test AUROC block and the **gate-cost table** (at 95/98/100%
  sensitivity, what fraction of healthy/sick does the gate reject?). The gate
  can only ever *lose* recall for the pipeline, so that table — not accuracy —
  is what goes in the report.
- **K2, K3** → two `sc_` mosaic_mixup cells at w500, on the sealed 360, key
  `lesion`, 179 GT boxes. ⚠ Never put these on the 142-box Active board, and
  never table them beside a CV fold number — three different scoreboards.
