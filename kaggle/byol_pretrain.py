"""
byol_pretrain.py — Self-supervised (BYOL) pretrain of the YOLOv8n BACKBONE on
                   unlabeled chest X-rays (Kaggle, NIH ChestX-ray14).

WHY
---
The TBX11K detector is data-limited: 799 labelled TB positives, and the wall is
RECALL. Every in-model and synthetic-augmentation lever has closed (resolution,
capacity, TTA, batch, all three copy-paste axes). The literature (and our own
copy-paste result) says generative synthesis cannot manufacture new clinical
variance beyond a duplicate-real baseline. The one lever that adds REAL
information is the huge pool of UNLABELLED in-domain chest X-rays we don't use.

BYOL (Bootstrap Your Own Latent) is a non-contrastive SSL method: an online net
predicts a slow EMA "target" net's embedding of a second augmented view of the
SAME image. No negatives, no labels — only images. We pretrain the YOLOv8n
backbone here, then load it as the TBX11K detector init (see
../yolo_experiments/exp11_byol_init.py) exactly like the VinDr supervised
backbone (weights/pretrain/vindr_supervised_best.pt). The clean headline is a
3-way INIT ablation on the frozen split:
    COCO-init 0.726  ·  VinDr (supervised transfer) 0.745  ·  BYOL (this, SSL)

The output we KEEP is a yolov8n-loadable `.pt` whose backbone is BYOL-adapted;
the neck/head stay at their COCO values so it drops straight into `YOLO(path)`.

SELF-CONTAINED on purpose (does NOT import yolo_common — unavailable on Kaggle),
mirroring kaggle/vindr_pretrain.py. The ONE non-negotiable invariant: the
architecture MUST stay yolov8n, or the backbone weights won't transfer.

DESIGN NOTES (each is a decision, not an accident)
-------------------------------------------------
- ENCODER = the actual yolov8n backbone (DetectionModel.model[0:10], modules
  Conv..SPPF, all `from=-1` so they run as a plain Sequential) + global avg pool.
  These are the SAME nn.Module objects living inside the full detector, so after
  training we save the whole detector and the backbone is already updated.
- INPUT REGIME = raw [0,1] (ToTensor), NO ImageNet mean/std. Ultralytics feeds
  the backbone [0,1] tensors, so SSL must match or fine-tuning sees a shifted
  input distribution.
- CXR-AWARE AUGS. hue/saturation are meaningless on grayscale (the MoCo-CXR
  lesson); we use random-resized-crop + h-flip + gaussian blur + mild
  brightness/contrast + (light) solarize, in the standard BYOL asymmetric pair.
  No vertical flip / heavy rotation — a chest film has a fixed upright pose.
- SSL_IMGSZ default 512 — MATCHES the detector. The conv backbone would transfer
  from 224 (ImageNet-style), but TB's wall is small/medium-lesion recall, and
  downsampling CXRs to 224 during SSL blurs exactly the fine opacities this
  pretraining is meant to teach. Matching 512 also removes a resolution-shift
  confound so the BYOL-init ablation is a PURE init comparison. Cost is ~5× the
  pixels of 224 → for the huge 112k FOUNDATION run, drop SSL_IMGSZ=256/384 for
  feasibility (the later VinDr@512 + TBX@512 stages re-introduce the fine scale);
  keep 512 for the smaller in-domain runs where it's cheap. Blur kernel auto-scales.
- AMP ON here (speed) — this is pretraining, NOT the deterministic detection
  floor where AMP is a confound. Seeded, but not bit-exact by design.
- COLLAPSE GUARD: BYOL's known failure is representation collapse. We log the
  mean per-dim std of the L2-normalised online projections (`emb_std`/`p_std`)
  each epoch; if it trends to ~0 the run has collapsed. Healthy for 256-d is
  ~1/sqrt(256) = 0.0625, and an untrained-encoder reference is measured at step 0.
  ⚠ Read `p_std`, NEVER the raw `z_enc_std`: the collapse is DIRECTIONAL (the
  encoder learns c(x)·v, one direction with a per-image magnitude), so the raw
  encoder spread EXPLODES while every normalised embedding converges to a point.
  Judging on z_enc_std reports a dying run as healthy — that mistake was made once.
- WHY THE 112k RUN COLLAPSED (probe, 2026-07-27; see yolo_experiments/byol_probe.py):
  the target-BN deviation, i.e. TARGET_BN_MODE. NOT the hyperparameters — arm A2
  (LR 1e-3→3e-4 plus grad clipping) still collapsed, so "raise LR-warmup" was the
  wrong instinct. The default is now `train`; VICReg (OBJECTIVE=vicreg) held at 122%
  in the same probe and is the pre-committed fallback if BYOL ever collapses again.

INIT SOURCE (MODEL= — this is the "start BYOL from tuned weights" option)
-------------------------------------------------------------------------
  MODEL=yolov8n.pt                              → BYOL-from-COCO  (SSL alone; the
                                                   clean "what does in-domain SSL add"
                                                   baseline vs COCO 0.726)
  MODEL=weights/pretrain/vindr_supervised_best.pt → VinDr→BYOL (supervised domain
                                                   pretrain THEN self-supervised
                                                   refinement — the best shot at
                                                   beating VinDr's 0.745; tests
                                                   whether SSL COMPOUNDS on an
                                                   already-tuned prior).
  The init source is stamped into the output filename (…_from{coco|vindr}_…), so the
  two variants never collide and exp11 can pick either with --model.
  ⚠ VinDr→BYOL should REFINE, not relearn: use a GENTLER LR and FEWER epochs
    (e.g. LR=3e-4 EPOCHS=40) so the box-agnostic BYOL objective doesn't wash out the
    supervised VinDr features (catastrophic forgetting). Attach vindr_supervised_best.pt
    to the Kaggle notebook (upload it as a private dataset) so MODEL= can find it.

2×T4 + MULTI-SESSION (Kaggle gives two T4s and 12h/session, ~30h/week)
---------------------------------------------------------------------
  ⚠ MULTI_GPU=1 (DataParallel) HAS A HOST-RAM LEAK — do NOT use it. An isolation matrix
    (2026-07-26, N_IMAGES=4000 toggling GPU_AUG × MULTI_GPU) proved nn.DataParallel leaks
    ~1 MB/step of host RAM (it re-replicates the whole module every forward), which OOM-kills
    the 112k run at epoch ~16. Single-GPU is dead-flat. AND for yolov8n @ batch 64 DP gives
    ~ZERO speedup (its scatter/gather overhead cancels the 2-GPU gain: 22.7 s/epoch single-GPU
    == 22.7 s/epoch DP on the probe). So:
  → Run SINGLE-GPU (MULTI_GPU=0, the default). No leak, same speed; batch 64 fits one T4 (~3.6 G).
  → To USE both T4s: launch TWO independent single-GPU jobs (DEVICE=cuda:0 and DEVICE=cuda:1) —
    e.g. different init sources / seeds — full 2× throughput, zero DP overhead, zero leak.
  → If you ever truly need one job across both cards, use DistributedDataParallel (persistent
    per-GPU replicas → no per-forward re-replicate → no leak), NOT DataParallel. Not needed here.
  SAVE_EVERY=5 → every 5 epochs write a resumable `<out>_<src>_state.pt` (optimizer
                 + target + epoch/step) AND a loadable `<out>_<src>_latest.pt`.
  RESUME=/kaggle/working/byol_yolov8n_coco_state.pt  → continue a run in the NEXT
                 session exactly where it stopped. Re-attach the previous session's output.
  Single-T4, 512/112k is ~11 min/epoch → size EPOCHS to fit a 12h session (or RESUME across).

GPU_AUG (the REAL throughput fix — Kaggle only gives 4 slow CPU cores)
---------------------------------------------------------------------
  GPU_AUG=1  → the CPU workers do ONLY decode+resize (cheap); all the random aug
               (crop/flip/jitter/blur) runs BATCHED ON THE GPU via kornia, under
               no_grad (random, NOT learnable). This moves the bottleneck off the
               4 saturated cores onto the idle GPUs → ~75 → ~10 min/epoch on 112k.
               Needs `pip install kornia`. Pair with PRECACHE so decode is cheap too.
  GPU_BASE=768 (default = CACHE_SIZE) → size the loader returns; the GPU RRC then
               crops to SSL_IMGSZ (512). 768 base is fully quality-neutral.

PRECACHE (the throughput unlock — Kaggle's dataset mount is slow to read)
------------------------------------------------------------------------
  PRECACHE=1  → before training, resize the whole corpus ONCE to CACHE_SIZE and
                write it to fast local disk (WORK_ROOT/cxr_cache_<size>), then read
                from there. Costs ~one slow pass up front (threaded), then every
                epoch is compute-bound instead of mount-bound (~78 → ~6-8 min/epoch
                on 112k). Idempotent: a resumed session re-uses whatever's cached.
  CACHE_SIZE=512  → cache resolution. 512 slightly upsamples the strongest crops
                (immaterial for SSL on a proxy corpus); 768 is fully quality-neutral.
  Note: the cache lives in /kaggle/working, which does NOT persist across sessions —
  a new session re-runs the (threaded, ~15-20 min) precache before training.

SCOPE KNOBS (the plan runs BOTH in parallel)
--------------------------------------------
  N_IMAGES=40000  EPOCHS=60   → the scoped PoC that FINISHES in-window
  N_IMAGES=0      EPOCHS=200  → the full 112k foundation run (0 = use all)

LEAK ISOLATION (diagnosing the ~epoch-16 host-OOM — a monotonic ~1 MB/step main-proc leak)
------------------------------------------------------------------------------------------
  mem_report() now also prints sysRAM=used/total (from /proc/meminfo) — TOTAL RAM over every
  process incl. the DataLoader workers, so a worker-side leak (invisible to /proc/self hostRSS)
  still shows. MEM_EVERY=20 prints RSS+sysRAM every N steps → read the intra-epoch slope directly.
  Fast bisect on a small corpus so epochs are ~20-30 s and the slope shows in 2-3 epochs:
    N_IMAGES=4000 EPOCHS=6 BATCH=64 PRECACHE=1 CACHE_SIZE=512 MEM_EVERY=20
  toggling ONE variable at a time — the config where the per-step slope goes FLAT is the culprit:
    A  GPU_AUG=1 MULTI_GPU=1  → reproduce the leak on the subset (baseline)
    B  GPU_AUG=0 MULTI_GPU=1  → flat ⇒ kornia GPU-aug is the leak
    C  GPU_AUG=1 MULTI_GPU=0  → flat ⇒ nn.DataParallel is the leak
  TRACEMALLOC=1 additionally prints the top Python allocation sites each epoch (Python-only —
  won't see C/CUDA/kornia-C++ allocations, so treat a flat tracemalloc as inconclusive, not clean).

RUN (Kaggle Notebook, GPU T4, Internet ON first run so YOLO can fetch yolov8n.pt)
--------------------------------------------------------------------------------
  1. Add data: NIH "nih-chest-xrays/data" (attaches under /kaggle/input).
  2. One cell:  %run byol_pretrain.py        (or paste this file into a cell)
  3. Knobs are env vars — set them in a cell before %run, e.g.:
        import os; os.environ["N_IMAGES"]="40000"; os.environ["EPOCHS"]="60"
  4. Output: /kaggle/working/<OUT_NAME>.pt  → download → drop into the repo at
     weights/pretrain/  and point exp11 at it.
  PLUMBING (no GPU, tiny, proves the wiring): SMOKE=1  (2 epochs, 64 imgs).
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CONFIG (all env-overridable; plain defaults) ──────────────────────────────
# ── Which backbone gets BYOL-adapted ──────────────────────────────────────────
# yolov8n  — the original arm. 1.27M params / 256-d. FAR outside BYOL's validated
#            envelope, and it collapsed twice before the target-BN bug was found.
# resnet50 — the ResNet-50 trunk out of torchvision's COCO-detection RetinaNet.
#            23.5M params / 2048-d = BYOL's REFERENCE configuration exactly, and the
#            same scaffolding BarlowTwins-CXR used successfully on this very corpus
#            (NIH 112k -> VinDr -> detection). Feeds yolo_common/tv_detect.py.
ARCH = os.environ.get("ARCH", "yolov8n").lower()
if ARCH not in ("yolov8n", "resnet50"):
    raise SystemExit(f"ARCH must be yolov8n|resnet50, got {ARCH!r}")

# Whose BYOL mechanics to use.
#   lightly  — the maintained reference implementation (pip install lightly). Its
#              BYOLProjectionHead/BYOLPredictionHead, NegativeCosineSimilarity and
#              update_momentum replace exactly the code that failed twice here.
#   internal — the hand-rolled version, kept so the yolov8n history stays reproducible.
# Verified against lightly 1.5.26 source: its BYOLProjectionHead defaults are
# (2048, 4096, 256) — identical to this arm's config — and there is NO .eval() anywhere
# in its models package, so its target network runs on BATCH statistics. That is an
# independent confirmation of TARGET_BN_MODE=train (our probe found it empirically).
BYOL_IMPL = os.environ.get("BYOL_IMPL", "lightly" if ARCH == "resnet50" else "internal").lower()
if BYOL_IMPL not in ("lightly", "internal"):
    raise SystemExit(f"BYOL_IMPL must be lightly|internal, got {BYOL_IMPL!r}")

# 512 matches the detector, but ResNet-50 is ~18x the FLOPs of the yolov8n backbone:
# measured on a 3050, 224=25 / 256=32 / 512=125 min per 112k epoch, so 30 epochs @512
# is ~31 h on a T4 and CANNOT finish a 12 h Kaggle session. 256 is the resnet50 default
# (~7 h for 30 epochs). This is standard practice, not a compromise — BYOL/SimCLR/MoCo/
# BarlowTwins all pretrain at 224 and transfer to detection at much higher resolution;
# SSL gains far more from epochs x data than from pretraining resolution. The supervised
# VinDr and TBX stages still run at 512. See ETA guard + BN_RECALIB in tv_detect.
SSL_IMGSZ = int(os.environ.get("SSL_IMGSZ", "256" if ARCH == "resnet50" else "512"))
EPOCHS    = int(os.environ.get("EPOCHS", "100"))
BATCH     = int(os.environ.get("BATCH", "64"))         # nano @512 fits a T4; 6GB local → 16
N_IMAGES  = int(os.environ.get("N_IMAGES", "0"))       # cap the corpus (0 = use all)
LR        = float(os.environ.get("LR", "1e-3"))        # AdamW; scaled for our batch
WD        = float(os.environ.get("WD", "1e-6"))
WARMUP_EP = int(os.environ.get("WARMUP_EP", "5"))
TAU_BASE  = float(os.environ.get("TAU_BASE", "0.996")) # EMA momentum → 1.0 on cosine
# BYOL's reference hidden width is 4096 for a 2048-d encoder. Keeping 2048 for the
# yolov8n arm preserves that arm's historical behaviour bit-for-bit.
PROJ_HID  = int(os.environ.get("PROJ_HID", "4096" if ARCH == "resnet50" else "2048"))
PROJ_OUT  = int(os.environ.get("PROJ_OUT", "256"))
DEVICE    = os.environ.get("DEVICE", "cuda:0")         # "cuda:0" | "cpu"; DP if MULTI_GPU=1
MULTI_GPU = os.environ.get("MULTI_GPU", "0") == "1"    # ⚠ DataParallel LEAKS host RAM (~1MB/step → OOM @ep16) AND ~0 speedup for yolov8n@b64 — keep 0; for 2×T4 run two single-GPU jobs (see docstring)
WORKERS   = int(os.environ.get("WORKERS", "4"))
PREFETCH  = int(os.environ.get("PREFETCH", "2"))       # batches prefetched per worker
BLUR_MAX  = int(os.environ.get("BLUR_MAX", "9"))       # cap the blur kernel (51 was absurd)
PRECACHE  = os.environ.get("PRECACHE", "0") == "1"     # resize corpus → fast local disk once
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "512"))  # cache resolution (768 = quality-neutral)
GPU_AUG   = os.environ.get("GPU_AUG", "0") == "1"      # do crop/flip/jitter/blur on GPU (kornia)
GPU_BASE  = int(os.environ.get("GPU_BASE", str(CACHE_SIZE)))  # size the loader returns (GPU crops to SSL_IMGSZ)
SEED      = int(os.environ.get("SEED", "0"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "0"))    # checkpoint every N epochs (0=end only)
RESUME    = os.environ.get("RESUME", "").strip()       # path to a *_state.pt to continue from
AMP       = os.environ.get("AMP", "1") == "1"          # mixed precision (speed); off for CPU
SMOKE     = os.environ.get("SMOKE", "0") == "1"        # 2 epochs, 64 imgs, plumbing only
MEM_EVERY   = int(os.environ.get("MEM_EVERY", "0"))    # >0: print RSS+sysRAM every N steps (leak isolation)
TRACEMALLOC = os.environ.get("TRACEMALLOC", "0") == "1"  # per-epoch top Python allocators (leak hunt; Python-only)

# ── Stability / probe knobs (added 2026-07-27 after the collapse→NaN 112k run) ─
# The 30-epoch run collapsed from epoch 1 (emb_std 0.034 → 0.0012 by ep4), then
# NaN'd at ep10 and burned 20 more epochs. These knobs were added to DIAGNOSE that,
# and every default reproduced the shipped behaviour while the probe was pending.
# The probe has now run (2026-07-27) and named the cause, so ONE default has moved:
# TARGET_BN_MODE. The rest still reproduce the old script. See
# yolo_experiments/byol_probe.py for the arms that use them.
OBJECTIVE      = os.environ.get("OBJECTIVE", "byol").lower()        # byol | vicreg
# ⚠ DEFAULT CHANGED 2026-07-28: eval → train. `eval` is the ~4-line deviation from
# reference BYOL that the probe proved COLLAPSES the representation (arm A0: p_std
# 1% of reference; arm A1 `train`: 113%, held 2975 steps). The 8.8 h / 112k run that
# produced a 100%-NaN backbone ran on the old default. `eval` is still reachable —
# byol_probe.py's A0 arm pins it explicitly to reproduce the failure.
TARGET_BN_MODE = os.environ.get("TARGET_BN_MODE", "train").lower()  # train (correct) | eval (collapses)
GRAD_CLIP      = float(os.environ.get("GRAD_CLIP", "0"))            # 0 = off (shipped)
PROBE_LOG      = int(os.environ.get("PROBE_LOG", "0"))              # >0: per-N-step diagnostics
MAX_STEPS      = int(os.environ.get("MAX_STEPS", "0"))              # >0: stop after N steps AND drive
                                                                    #     the LR/tau schedule off it
WARMUP_STEPS   = int(os.environ.get("WARMUP_STEPS", "0"))           # >0: overrides WARMUP_EP (probe)
# Collapse / NaN halts — these do NOT prevent collapse, they stop us paying 8h for it.
COLLAPSE_HALT     = os.environ.get("COLLAPSE_HALT", "1") == "1"
COLLAPSE_FLOOR    = float(os.environ.get("COLLAPSE_FLOOR", "0.25"))  # frac of the step-0 reference
COLLAPSE_PATIENCE = int(os.environ.get("COLLAPSE_PATIENCE", "200"))  # consecutive steps under floor
# VICReg coefficients (paper defaults 25/25/1). Only used when OBJECTIVE=vicreg.
# ETA guard — project total wall-clock after a few steps and warn if it cannot fit a
# Kaggle session. Cheap insurance against discovering a 30 h config at hour 12.
ETA_AFTER_STEPS = int(os.environ.get("ETA_AFTER_STEPS", "30"))
SESSION_HOURS   = float(os.environ.get("SESSION_HOURS", "11"))   # Kaggle caps at 12
VIC_SIM = float(os.environ.get("VIC_SIM", "25.0"))
VIC_STD = float(os.environ.get("VIC_STD", "25.0"))
VIC_COV = float(os.environ.get("VIC_COV", "1.0"))

INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input"))
WORK_ROOT  = Path(os.environ.get("WORK_ROOT", "/kaggle/working"))
# Name the artifact after the ARCH it actually contains — a resnet50 trunk written as
# "byol_yolov8n_*.pt" is a filename that lies, and exp11's auto-glob matches byol_*.pt.
OUT_NAME   = os.environ.get("OUT_NAME", f"byol_{ARCH}")
BASE_MODEL = os.environ.get("MODEL", "yolov8n.pt")     # MUST stay yolov8n for transfer

IMG_EXTS = (".png", ".jpg", ".jpeg")


# ── Corpus discovery (don't hardcode a dataset slug) ──────────────────────────
def index_images() -> list[Path]:
    """Every image under INPUT_ROOT (recursively). NIH ships them as
    images_0XX/images/*.png — rglob finds them all regardless of layout."""
    paths = [p for p in INPUT_ROOT.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not paths:
        raise SystemExit(
            f"HALT: no images under {INPUT_ROOT}. Attach the NIH ChestX-ray14 "
            f"dataset (nih-chest-xrays/data) — or any unlabelled CXR image set.")
    paths.sort()
    rng = random.Random(SEED)
    rng.shuffle(paths)
    if SMOKE:
        paths = paths[:64]
    elif N_IMAGES > 0:
        paths = paths[:N_IMAGES]
    print(f"[corpus] {len(paths)} images (of the pool under {INPUT_ROOT})")
    return paths


def precache_corpus(paths, cache_dir: Path, size: int):
    """One-time: resize every corpus image to `size` and write it to fast local
    disk (`cache_dir`), so training reads small local PNGs instead of big images
    off Kaggle's slow dataset mount. I/O-bound → threaded. IDEMPOTENT (skips files
    already cached), so a resumed session re-uses whatever is already there.

    Quality note: `size` must be ≥ the largest crop RandomResizedCrop takes (with
    scale.min=0.4 that's ~0.63·side), so size 512 slightly upsamples the strongest
    crops and 768 is fully quality-neutral. Fine either way for SSL on a proxy corpus.
    """
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(p):
        out = cache_dir / f"{p.stem}.png"
        if out.exists():
            return out
        try:
            Image.open(p).convert("RGB").resize((size, size)).save(out)
            return out
        except Exception as e:
            print(f"  [precache] skip {p.name}: {e}", flush=True)
            return None

    cached, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:      # threads hide mount I/O latency
        for i, r in enumerate(ex.map(one, paths)):
            if r is not None:
                cached.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  [precache] {i+1}/{len(paths)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[precache] {len(cached)} imgs @ {size}px → {cache_dir}  ({time.time()-t0:.0f}s, "
          f"one-time; subsequent epochs read from here)", flush=True)
    return cached


# ── Two-view BYOL transform (CXR-aware) ───────────────────────────────────────
def build_transforms():
    from torchvision import transforms as T

    # Blur kernel: sigma≤2 needs only ~13 taps; a 0.1×512=51 kernel is ~4x the cost
    # for no benefit and was starving the GPU. Cap it (BLUR_MAX) and keep it odd.
    kb = min(int(0.1 * SSL_IMGSZ), BLUR_MAX)
    kb = max(3, kb + 1 if kb % 2 == 0 else kb)

    # Grayscale CXR: NO hue/saturation. Brightness/contrast only, plus blur and a
    # light solarize on the second view (standard BYOL asymmetry).
    base = [
        T.RandomResizedCrop(SSL_IMGSZ, scale=(0.4, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
    ]
    view1 = T.Compose(base + [
        T.RandomApply([T.GaussianBlur(kernel_size=kb, sigma=(0.1, 2.0))], p=1.0),
        T.ToTensor(),                                   # → [0,1], 3ch (image is RGB)
    ])
    view2 = T.Compose(base + [
        T.RandomApply([T.GaussianBlur(kernel_size=kb, sigma=(0.1, 2.0))], p=0.1),
        T.RandomSolarize(threshold=0.5, p=0.2),         # applied on the PIL image
        T.ToTensor(),
    ])
    return view1, view2


class TwoViewCXR:
    """Loads a CXR as 3-channel RGB and returns two independently-augmented views."""
    def __init__(self, paths, view1, view2):
        import numpy as np
        # NOT a Python list: with fork DataLoader workers, refcount churn on a big
        # list of Path objects triggers copy-on-write growth → RAM leak → OOM after
        # ~16 epochs on 112k. A numpy string array is a single object → no per-item
        # refcounting in the workers. (This is what OOM-killed the first full run.)
        self.paths = np.array([str(p) for p in paths])
        self.view1, self.view2 = view1, view2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        # NIH PNGs are mostly 8-bit grayscale; a few are RGBA. Force 3ch RGB so
        # the yolov8 stem (3-in) is happy and R=G=B (true grayscale).
        img = Image.open(str(self.paths[i])).convert("RGB")
        return self.view1(img), self.view2(img)


class RawCXR:
    """GPU-aug mode: the CPU workers do only the cheap part — decode + resize to
    `base` + a uint8 CHW tensor. All the expensive random aug (crop/flip/jitter/
    blur) then runs BATCHED ON THE GPU (kornia), off Kaggle's 4 saturated cores.
    uint8 (not float) also quarters the CPU→GPU transfer and the worker shm."""
    def __init__(self, paths, base):
        import numpy as np
        self.paths = np.array([str(p) for p in paths])   # numpy, NOT a list — see TwoViewCXR
        self.base = base

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import numpy as np
        from PIL import Image
        img = Image.open(str(self.paths[i])).convert("RGB")
        if img.size != (self.base, self.base):
            img = img.resize((self.base, self.base))
        arr = np.array(img, dtype=np.uint8)                       # HWC — np.array COPIES (writable);
        #   np.asarray gave a read-only view of PIL's buffer → torch "not writable" UserWarning spam.
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8


def build_gpu_augs():
    """Two kornia AugmentationSequential pipelines (the BYOL asymmetric pair) that
    run on GPU tensor batches. Same recipe as the CPU transforms: RRC→512, h-flip,
    brightness/contrast (grayscale — no hue/sat), gaussian blur, + light solarize
    on view 2. Applied under no_grad — random, NOT learnable."""
    import kornia.augmentation as K
    kb = min(int(0.1 * SSL_IMGSZ), BLUR_MAX)
    kb = max(3, kb + 1 if kb % 2 == 0 else kb)

    def seq(blur_p, solarize):
        mods = [K.RandomResizedCrop((SSL_IMGSZ, SSL_IMGSZ), scale=(0.4, 1.0)),
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(0.4, 0.4, 0.0, 0.0, p=0.8),
                K.RandomGaussianBlur((kb, kb), (0.1, 2.0), p=blur_p)]
        if solarize:
            mods.append(K.RandomSolarize(0.1, p=0.2))
        return K.AugmentationSequential(*mods, same_on_batch=False)

    return seq(1.0, False), seq(0.1, True)


# ── Model: yolov8n backbone encoder + BYOL heads ──────────────────────────────
def mlp(in_dim, hid, out):
    return nn.Sequential(
        nn.Linear(in_dim, hid, bias=False), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
        nn.Linear(hid, out),
    )


class Enc(nn.Module):
    """DetectionModel.model[0:10] (Conv..SPPF) + global avg pool → (B, C) feature.
    The backbone slice shares module refs with the full detector, so training
    these params updates the detector in place (that's what we save downstream)."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for m in self.backbone:              # all from=-1 → pure sequential
            x = m(x)
        return torch.flatten(self.pool(x), 1)


def proj_first_linear(proj):
    """The projector's first Linear, for the bn_var_min diagnostic.

    The internal head is a bare nn.Sequential (index [0]); lightly's ProjectionHead
    wraps its Sequential in `.layers`. Indexing the wrong one silently probes the
    wrong tensor — or raises — so resolve it explicitly rather than assuming.
    """
    inner = getattr(proj, "layers", proj)
    return inner[0]


def probe_feat_dim(enc) -> int:
    """Feature width, WITHOUT disturbing the encoder.

    ⚠ This must run in eval(): a train-mode forward updates every BatchNorm's
    running_mean/running_var, so probing with a zeros tensor silently corrupts the
    init's BN statistics before step 0 — and, worse, makes a completely untrained
    encoder differ from its own source weights, which defeats the "did the encoder
    actually move" guard in save_resnet50_init. Caught 2026-07-28 by that guard's
    negative test reporting L2 7.28 on an untrained net. Same discipline the
    probe_stats() instrumentation already follows.
    """
    was_training = enc.training
    enc.eval()
    try:
        with torch.no_grad():
            return enc(torch.zeros(1, 3, SSL_IMGSZ, SSL_IMGSZ)).shape[1]
    finally:
        enc.train(was_training)


def make_backbone_encoder(detmodel):
    enc = Enc(detmodel.model[:10])
    return enc, probe_feat_dim(enc)


def make_resnet50_encoder():
    """ResNet-50 trunk lifted out of torchvision's COCO-DETECTION RetinaNet + GAP.

    Why the detection weights and not ImageNet: the downstream baseline we have to
    beat is that exact RetinaNet at Active mAP50 0.739, whose backbone/FPN/heads are
    all COCO-detection-trained. Starting BYOL from the same trunk keeps the ablation
    to ONE variable — "did in-domain SSL help THIS model" — rather than confounding
    it with a change of pretraining corpus. (BarlowTwins-CXR started from ImageNet;
    set INIT=imagenet to mirror the paper instead.)

    Two things verified locally before this was written, because both are classic
    torchvision traps:
      * `retinanet_resnet50_fpn_v2` uses REAL BatchNorm2d, not FrozenBatchNorm2d, so
        the encoder has live BN dynamics — which BYOL's collapse resistance needs.
        (The v1 detection models DO use FrozenBN. Do not swap the arch casually.)
      * detection backbones ship with `trainable_backbone_layers=3`, i.e. conv1/bn1/
        layer1 arrive with requires_grad=False (126/159 tensors trainable). SSL must
        train the WHOLE trunk, so grad is re-enabled below. This is the same class of
        bug as the 2026-07-26 yolov8n freeze that silently produced COCO inits.
    Round-trip back into `det.backbone.body` is exact: 318/318 tensors, 0 missing.
    """
    from torchvision.models import resnet50
    init = os.environ.get("INIT", "coco").lower()
    if init == "coco":
        from torchvision.models.detection import (
            retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights)
        det = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        r = resnet50(weights=None)
        missing, unexpected = r.load_state_dict(det.backbone.body.state_dict(), strict=False)
        missing = [m for m in missing if not m.startswith("fc.")]      # fc is unused
        if missing or unexpected:
            raise SystemExit(f"HALT: COCO trunk did not map cleanly onto resnet50 — "
                             f"missing={missing[:5]} unexpected={unexpected[:5]}")
        del det
    elif init == "imagenet":
        from torchvision.models import ResNet50_Weights
        r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif init == "random":
        r = resnet50(weights=None)
    else:
        raise SystemExit(f"INIT must be coco|imagenet|random, got {init!r}")

    trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                          r.layer1, r.layer2, r.layer3, r.layer4)
    enc = nn.Sequential(trunk, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
    enc.requires_grad_(True)                     # see docstring: detection default freezes early layers
    feat_dim = probe_feat_dim(enc)
    print(f"[cfg] encoder = resnet50 ({init}-init), feat_dim={feat_dim}, "
          f"{sum(p.numel() for p in enc.parameters()):,} params")
    return enc, feat_dim


def vicreg_loss(z1, z2, sim_c=25.0, std_c=25.0, cov_c=1.0):
    """VICReg (Bardes et al.) — the non-contrastive objective WITHOUT BYOL's implicit
    collapse dynamics. Three terms: invariance (MSE between views), variance (a HINGE
    that penalises any per-dim std below 1) and covariance (decorrelates dims).

    WHY IT'S HERE: the variance hinge is an EXPLICIT loss term against exactly the
    failure we measured — per-dim std sliding to ~0. BYOL only resists collapse
    implicitly (predictor + EMA + batch stats), which was evidently too weak for a
    1.27M-param / 256-d encoder at batch 64. VICReg keeps every property we want from
    BYOL: non-contrastive, NO negatives (so no false-negative problem on a corpus that
    is ~54% normal CXR), two views, augmentation-robust.

    Computed in fp32 — the covariance term is a D×D outer product and is not
    fp16-safe. Returns a SCALAR (a batch-level statistic, unlike BYOL's per-sample loss).
    """
    z1, z2 = z1.float(), z2.float()
    N, D = z1.shape
    repr_loss = F.mse_loss(z1, z2)                       # invariance

    z1c, z2c = z1 - z1.mean(0), z2 - z2.mean(0)
    std1 = torch.sqrt(z1c.var(dim=0) + 1e-4)             # variance HINGE at 1.0
    std2 = torch.sqrt(z2c.var(dim=0) + 1e-4)
    std_loss = 0.5 * (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean())

    def off_diag_sq(zc):                                 # covariance, off-diagonal only
        cov = (zc.T @ zc) / max(1, N - 1)
        return (cov.pow(2).sum() - cov.pow(2).diagonal().sum()) / D
    cov_loss = off_diag_sq(z1c) + off_diag_sq(z2c)

    return sim_c * repr_loss + std_c * std_loss + cov_c * cov_loss


class BYOLNet(nn.Module):
    """The ENTIRE BYOL step as one nn.Module so DataParallel can split a batch
    across both T4s (encoder + proj + pred + target, not just the encoder).
    forward(v1,v2) → (per-sample loss (b,), normalised online pred (b,d)). AMP is
    enabled INSIDE forward — DataParallel spawns worker threads and autocast is
    thread-local, so an outer `with autocast` would NOT reach the replicas."""
    def __init__(self, detmodel=None):
        super().__init__()
        # ARCH picks the encoder; everything downstream of it is architecture-agnostic
        # (same aug pipeline, same loop, same guards, same collapse metric).
        if ARCH == "resnet50":
            self.online_enc, feat = make_resnet50_encoder()
        else:
            self.online_enc, feat = make_backbone_encoder(detmodel)
        if BYOL_IMPL == "lightly":
            try:
                from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
            except ImportError as e:
                raise SystemExit(
                    f"BYOL_IMPL=lightly needs the reference implementation: "
                    f"`pip install lightly`  ({e})")
            # Same geometry as lightly's own defaults for a 2048-d encoder.
            self.online_proj = BYOLProjectionHead(feat, PROJ_HID, PROJ_OUT)
            self.online_pred = BYOLPredictionHead(PROJ_OUT, PROJ_HID, PROJ_OUT)
        else:
            self.online_proj = mlp(feat, PROJ_HID, PROJ_OUT)
            self.online_pred = mlp(PROJ_OUT, PROJ_HID, PROJ_OUT)
        # CRITICAL: Ultralytics loads yolov8n.pt with requires_grad=False on EVERY
        # param (inference default). The backbone slice inherits that, so without
        # re-enabling grad the optimizer (built from online_params = requires_grad
        # params) trains ONLY the proj/pred heads and the ENCODER stays frozen at
        # COCO — silently defeating BYOL (the saved backbone == COCO; BN stats still
        # move on forward, which masks it). Enable grad on the whole online tower
        # BEFORE deep-copying the EMA target (which is then re-frozen below).
        for p in self.online_enc.parameters():
            p.requires_grad_(True)
        for p in self.online_proj.parameters():
            p.requires_grad_(True)
        for p in self.online_pred.parameters():
            p.requires_grad_(True)
        if BYOL_IMPL == "lightly":
            from lightly.loss import NegativeCosineSimilarity
            self._crit = NegativeCosineSimilarity()
        self.target_enc = copy.deepcopy(self.online_enc)     # EMA copies, no grad
        self.target_proj = copy.deepcopy(self.online_proj)
        for p in list(self.target_enc.parameters()) + list(self.target_proj.parameters()):
            p.requires_grad_(False)
        self.feat_dim = feat

    def set_modes(self):
        self.online_enc.train(); self.online_proj.train(); self.online_pred.train()
        # TARGET_BN_MODE: `train` (the default since 2026-07-28) matches reference BYOL —
        # the target's BN uses BATCH statistics, which is the mechanism BYOL's collapse
        # resistance is usually attributed to. `eval` (this script's original behaviour)
        # normalises with the RUNNING stats copied from online in update_target() instead,
        # which removes that pressure. The probe CONFIRMED this is the collapse: A0 (eval)
        # fell to 1% of the reference p_std, A1 (train) held at 113% for 2975 steps.
        # `eval` is kept only so byol_probe.py's A0 arm can reproduce the failure.
        tgt_train = (TARGET_BN_MODE == "train")
        self.target_enc.train(tgt_train); self.target_proj.train(tgt_train)

    def forward(self, v1, v2):
        with torch.autocast(device_type=v1.device.type, enabled=(AMP and v1.device.type == "cuda")):
            if OBJECTIVE == "vicreg":
                # No predictor, no EMA target — both views go through the SAME online tower.
                zv1 = self.online_proj(self.online_enc(v1))
                zv2 = self.online_proj(self.online_enc(v2))
                loss_s = vicreg_loss(zv1, zv2, VIC_SIM, VIC_STD, VIC_COV)
                # Per-sample shape so the DP gather + .mean() in the loop is unchanged;
                # mean of b identical copies == the scalar, and the gradient is correct.
                loss = loss_s.reshape(1).expand(v1.shape[0])
                # Monitor on the SAME scale as the BYOL arms (L2-normalised per-dim std)
                # so emb_std is comparable across arms; VICReg itself does not normalise.
                return loss, F.normalize(zv1.float(), dim=1).detach()
            p1 = self.online_pred(self.online_proj(self.online_enc(v1)))
            p2 = self.online_pred(self.online_proj(self.online_enc(v2)))
            with torch.no_grad():
                z1 = self.target_proj(self.target_enc(v1))
                z2 = self.target_proj(self.target_enc(v2))

            if BYOL_IMPL == "lightly":
                # lightly's NegativeCosineSimilarity reduces to a SCALAR (no reduction
                # arg). Expanded to (b,) so the loop's DP-gather + .mean() is unchanged —
                # the mean of b identical copies is the scalar, and the gradient is
                # correct. Same trick already used by the VICReg arm above.
                # ⚠ SCALE CHANGES vs the internal path: this is -cos ∈ [-1, 1] averaged
                # over the two directions, where internal used (2 - 2cos) summed ∈ [0, 8].
                # Same optimum, same gradient direction, different printed magnitude —
                # do NOT compare loss values across BYOL_IMPL. The collapse guard is on
                # p_std, not loss, so it is unaffected.
                loss_s = 0.5 * (self._crit(p1.float(), z2.float())
                                + self._crit(p2.float(), z1.float()))
                loss = loss_s.reshape(1).expand(v1.shape[0])
            else:
                def d(p, z):
                    p = F.normalize(p.float(), dim=1); z = F.normalize(z.float(), dim=1)
                    return 2 - 2 * (p * z).sum(dim=1)        # (b,)
                loss = d(p1, z2) + d(p2, z1)                 # (b,)
        return loss, F.normalize(p1.float(), dim=1).detach()  # detach: monitor only, no graph held


@torch.no_grad()
def update_target(net: BYOLNet, tau: float):
    """EMA the online → target. Runs on the MASTER net; DataParallel re-replicates
    from it each forward, so replicas stay in sync.

    ⚠ THE BUFFER QUESTION. The internal path also copies BN *buffers* online→target
    every step. lightly's `update_momentum` does NOT — it touches parameters only
    (verified in 1.5.26 source). With the target in train mode the copy is inert,
    because train-mode BN normalises with BATCH statistics and ignores the running
    buffers entirely. But under the OLD `TARGET_BN_MODE=eval` default those copied
    buffers WERE what the target normalised with — that copy was the second half of
    the collapse. The lightly path drops it to match the reference exactly.
    """
    if BYOL_IMPL == "lightly":
        from lightly.models.utils import update_momentum
        update_momentum(net.online_enc, net.target_enc, m=tau)
        update_momentum(net.online_proj, net.target_proj, m=tau)
        return
    for og, tg in ((net.online_enc, net.target_enc), (net.online_proj, net.target_proj)):
        for po, pt in zip(og.parameters(), tg.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
        for bo, bt in zip(og.buffers(), tg.buffers()):
            bt.copy_(bo)


def online_params(net: BYOLNet):
    return [p for p in net.parameters() if p.requires_grad]


@torch.no_grad()
def probe_stats(net: BYOLNet, v1):
    """Diagnostics that localise WHERE a collapse starts. The per-epoch emb_std only
    sees the very END of the tower (predictor output), so it cannot distinguish "the
    encoder went constant" from "the heads degenerated" — that ambiguity is what made
    the 8.8h NaN run unreadable.

    Returns:
      z_enc_std   mean per-dim std of the RAW encoder output (pre-projector). THE key
                  signal: if this slides, the backbone itself is collapsing and the
                  saved init is worthless regardless of what the loss says.
      bn_var_min  smallest per-dim BATCH variance entering the projector's BatchNorm.
                  Tests the proposed NaN mechanism directly — as this → 0, BN divides
                  by ~0, activations explode, fp16 overflows, GradScaler underflows.
      z_proj_std  same measure at the projector output.

    Runs in eval() so it neither pollutes BN running stats nor perturbs training;
    restored via set_modes() by the caller. Costs one extra encoder pass, only on
    logged steps.
    """
    was_training = net.online_enc.training
    net.online_enc.eval(); net.online_proj.eval()
    z_enc = net.online_enc(v1).float()
    hidden = proj_first_linear(net.online_proj)(z_enc).float()   # the Linear feeding BatchNorm1d
    z_proj = net.online_proj(z_enc).float()
    out = {
        "z_enc_std": float(z_enc.std(dim=0).mean()),
        "bn_var_min": float(hidden.var(dim=0).min()),
        "z_proj_std": float(F.normalize(z_proj, dim=1).std(dim=0).mean()),
    }
    if was_training:
        net.online_enc.train(); net.online_proj.train()
    return out


def cosine(step, total, base, final=1.0):
    return final - (final - base) * (math.cos(math.pi * step / total) + 1) / 2


def mem_report() -> str:
    """host RSS (this proc) + TOTAL system RAM used + GPU alloc/reserved — printed each
    epoch (and every MEM_EVERY steps) to catch a leak in the act. sysRAM is the key one:
    it's every process incl. the DataLoader workers (a worker-side leak is INVISIBLE to
    /proc/self hostRSS), computed as MemTotal-MemAvailable from /proc/meminfo (no psutil dep)."""
    parts = []
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts.append("hostRSS=" + "".join(line.split()[1:3])); break
    except Exception:
        pass
    try:
        mi = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                mi[k] = int(rest.split()[0])                 # kB
        used = mi["MemTotal"] - mi.get("MemAvailable", mi["MemTotal"])
        parts.append(f"sysRAM={used//1024}/{mi['MemTotal']//1024}MB")
    except Exception:
        pass
    if torch.cuda.is_available():
        parts.append(f"gpu={torch.cuda.memory_allocated()/1e9:.2f}/"
                     f"{torch.cuda.memory_reserved()/1e9:.2f}G")
    return " ".join(parts)


def seed_everything():
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ── Save an Ultralytics-loadable yolov8n init (BYOL backbone + COCO neck/head) ─
def assert_finite(sd, what: str) -> None:
    """Refuse to persist a NaN/Inf artifact. The 2026-07-27 run wrote FIVE checkpoints
    after it had already gone NaN, each one overwriting the rolling `_latest.pt` — so
    the last finite state (epoch 5) was destroyed by its own successors. A checkpoint
    that cannot be trained from is worse than no checkpoint: it looks like progress."""
    bad = [k for k, v in sd.items()
           if torch.is_tensor(v) and torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(
            f"HALT: refusing to write {what} — {len(bad)}/{len(sd)} tensors are non-finite "
            f"(first: {bad[:3]}). Writing this would overwrite the last GOOD checkpoint with "
            f"an unusable one, which is how the 112k run lost every recoverable epoch.")


def save_resnet50_init(net, out_path: Path, meta: dict):
    """Persist the BYOL-adapted ResNet-50 trunk as a plain state_dict.

    NOT an Ultralytics checkpoint — this artifact feeds torchvision's RetinaNet via
    `yolo_common/tv_detect.py`, which loads it into `model.backbone.body`. The keys
    are plain-resnet50 names (conv1/bn1/layer1..layer4), which is exactly what that
    body expects; the round-trip is 318/318 tensors with 0 missing and 0 unexpected.
    `fc.*` is absent by construction (the trunk has no classifier).
    """
    import torch
    enc = net.online_enc                                   # Sequential(trunk, pool, flatten)
    trunk = enc[0]
    names = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    sd = {}
    for name, mod in zip(names, trunk):
        for k, v in mod.state_dict().items():
            # ⚠ Do NOT blanket-.float() here. `num_batches_tracked` is an int64 STEP
            # COUNTER, and casting it to float made it look like a weight to every
            # is_floating_point() test downstream — which silently reduced the
            # "did the encoder move" drift below to a step count (caught 2026-07-29:
            # the reported 74985.132 was exactly sqrt(53)*2*5150 steps).
            sd[f"{name}.{k}"] = v.detach().cpu().clone()
    assert_finite(sd, f"resnet50 trunk {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "resnet50", "format": "torchvision_resnet_trunk",
                "state_dict": sd, "byol_meta": meta,
                "date": datetime.now(timezone.utc).isoformat()}, out_path)
    # Prove it reloads INTO A REAL DETECTOR before we trust it downstream — the same
    # discipline as the yolov8n path's YOLO() reload check.
    from torchvision.models.detection import (
        retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights)
    det = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    coco_sd = {k: v.detach().cpu().clone() for k, v in det.backbone.body.state_dict().items()}
    missing, unexpected = det.backbone.body.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"HALT: saved trunk will not load into RetinaNet — "
                         f"missing={len(missing)} unexpected={len(unexpected)}")
    # PROVE THE ENCODER MOVED. The 2026-07-26 disaster was a frozen encoder writing
    # checkpoints byte-identical to the init for two whole runs, undetected because
    # the loss curve (heads fitting fixed features) and BN running stats both looked
    # alive. A load check alone would NOT have caught it — this L2 does.
    #
    # ⚠ WEIGHTS ONLY. Excluding `num_batches_tracked` is load-bearing, not tidiness: a
    # FROZEN encoder still runs forward passes, so that counter still ticks. Including
    # it made this metric nonzero for a frozen encoder — i.e. the guard would have
    # PASSED the exact bug it was written to catch. BN running buffers are reported
    # SEPARATELY for the same reason: "the encoder learned" must not be satisfiable by
    # BN statistics drifting while the weights sit still.
    drift = drift_bn = 0.0
    for k in sd:
        if k not in coco_sd or "num_batches_tracked" in k or not sd[k].is_floating_point():
            continue
        d = float((sd[k].double() - coco_sd[k].double()).pow(2).sum())
        if k.endswith("running_mean") or k.endswith("running_var"):
            drift_bn += d
        else:
            drift += d
    drift, drift_bn = drift ** 0.5, drift_bn ** 0.5
    if drift == 0.0 and meta.get("note") != "untrained":
        raise SystemExit(
            "HALT: the saved trunk's LEARNABLE weights are byte-identical to the COCO "
            "init — the encoder never trained. BN running stats may still have moved, "
            "which is exactly what masked this bug in 2026-07-26. Not an SSL result.")
    print(f"[save] wrote + RetinaNet-reload-verified {out_path} "
          f"({len(sd)} tensors, 0 missing/unexpected, "
          f"L2-vs-COCO: weights {drift:.3f}, bn-stats {drift_bn:.3f})")


def save_any(yolo, net, out_path: Path, meta: dict, dev=None) -> None:
    """ARCH dispatcher for the two artifact formats. Keeps the training loop from
    caring which backbone it just trained."""
    meta = {**meta, "arch": ARCH}
    if ARCH == "resnet50":
        save_resnet50_init(net, out_path, meta)
    else:
        save_init(yolo, out_path, meta)
        if dev is not None:
            yolo.model.to(dev)               # save_init moved it to CPU/float


def save_init(yolo, out_path: Path, meta: dict):
    import torch
    import ultralytics
    detmodel = yolo.model.float().cpu().eval()
    assert_finite(detmodel.state_dict(), f"detector init {out_path.name}")
    ckpt = {
        "epoch": -1,
        "best_fitness": None,
        "model": copy.deepcopy(detmodel).half(),   # loader .float()s it back
        "ema": None, "updates": None, "optimizer": None,
        "train_args": {}, "train_results": None,
        "date": datetime.now(timezone.utc).isoformat(),
        "version": ultralytics.__version__,
        "byol_meta": meta,                          # provenance, ignored by YOLO()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    # Prove it reloads as a detector before we trust it downstream.
    from ultralytics import YOLO
    _ = YOLO(str(out_path))
    print(f"[save] wrote + reload-verified {out_path}")


# ── Training-state checkpoint (RESUME across Kaggle's 12h sessions) ────────────
def save_state(path: Path, net: BYOLNet, opt, scaler, ep: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_finite(net.online_enc.state_dict(), f"resume state {path.name} (online_enc)")
    assert_finite(net.target_enc.state_dict(), f"resume state {path.name} (target_enc)")
    torch.save({
        "online_enc": net.online_enc.state_dict(),
        "online_proj": net.online_proj.state_dict(),
        "online_pred": net.online_pred.state_dict(),
        "target_enc": net.target_enc.state_dict(),
        "target_proj": net.target_proj.state_dict(),
        "optim": opt.state_dict(), "scaler": scaler.state_dict(),
        "epoch": ep, "step": step,
    }, path)
    print(f"[state] resume checkpoint → {path}  (epoch {ep}, step {step})")


def load_state(path: Path, net: BYOLNet, opt, scaler, dev):
    st = torch.load(path, map_location=dev, weights_only=False)
    net.online_enc.load_state_dict(st["online_enc"])
    net.online_proj.load_state_dict(st["online_proj"])
    net.online_pred.load_state_dict(st["online_pred"])
    net.target_enc.load_state_dict(st["target_enc"])
    net.target_proj.load_state_dict(st["target_proj"])
    opt.load_state_dict(st["optim"]); scaler.load_state_dict(st["scaler"])
    print(f"[resume] loaded {path} → continue at epoch {st['epoch']} (step {st['step']})")
    return st["epoch"], st["step"]


def main():
    import torch
    from torch.utils.data import DataLoader
    from ultralytics import YOLO

    seed_everything()
    if TRACEMALLOC:
        import tracemalloc; tracemalloc.start(10)          # per-epoch top Python allocators (leak hunt)
    dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    epochs = 2 if SMOKE else EPOCHS
    print(f"[cfg] imgsz={SSL_IMGSZ} epochs={epochs} batch={BATCH} lr={LR} "
          f"tau_base={TAU_BASE} device={dev} smoke={SMOKE}")

    paths = index_images()
    if PRECACHE:
        cd_env = os.environ.get("CACHE_DIR", "").strip()   # Path("") is truthy → check the str
        cache_dir = Path(cd_env) if cd_env else (WORK_ROOT / f"cxr_cache_{CACHE_SIZE}")
        print(f"[precache] enabled → resizing corpus to {CACHE_SIZE}px at {cache_dir}", flush=True)
        paths = precache_corpus(paths, cache_dir, CACHE_SIZE)
        if not paths:
            raise SystemExit("HALT: precache produced 0 images.")

    gpu_aug = None
    if GPU_AUG:
        print(f"[cfg] GPU augmentation ON (kornia) — CPU decodes @ {GPU_BASE}px, "
              f"GPU crops to {SSL_IMGSZ}px. Frees the 4 CPU cores.")
        ds = RawCXR(paths, GPU_BASE)
        gpu_aug = build_gpu_augs()
    else:
        view1, view2 = build_transforms()
        ds = TwoViewCXR(paths, view1, view2)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                        drop_last=True, pin_memory=False,      # OOM guards: no pin thread, and
                        persistent_workers=False,              # respawn workers each epoch → free
                        prefetch_factor=(PREFETCH if WORKERS > 0 else None))  # any per-worker RAM
    if len(loader) == 0:
        raise SystemExit(
            f"HALT: 0 batches — {len(ds)} images < batch {BATCH} with drop_last. "
            f"Lower BATCH (e.g. BATCH={min(16, max(2, len(ds)//2))}).")

    # Init source: COCO (yolov8n.pt) OR the VinDr supervised backbone → BYOL then
    # REFINES an already domain-tuned prior (supervised-then-SSL). Stamped into the
    # output name so the two variants never collide / are distinguishable downstream.
    if ARCH == "resnet50":
        src = os.environ.get("INIT", "coco").lower()          # coco | imagenet | random
        print(f"[cfg] arch = resnet50, init source = {src}")
        yolo = None                                           # no Ultralytics model in this arm
        net = BYOLNet().to(dev)
    else:
        src = ("vindr" if "vindr" in Path(BASE_MODEL).name.lower()
               else "coco" if BASE_MODEL == "yolov8n.pt" else Path(BASE_MODEL).stem)
        print(f"[cfg] init source = {src}  ({BASE_MODEL})")
        yolo = YOLO(BASE_MODEL)                  # backbone gets BYOL-adapted from here
        net = BYOLNet(yolo.model).to(dev)
    if gpu_aug is not None:
        gpu_aug = tuple(a.to(dev) for a in gpu_aug)
    opt = torch.optim.AdamW(online_params(net), lr=LR, weight_decay=WD)
    # GUARD: PROVE the ENCODER is actually being optimized. Ultralytics loads
    # yolov8n.pt frozen (requires_grad=False on all params); if the online encoder
    # isn't re-enabled the optimizer silently trains only the proj/pred heads and
    # the saved backbone stays COCO — a whole run wasted (this bit us once). Halt.
    n_enc = sum(p.requires_grad for p in net.online_enc.parameters())
    n_enc_params = sum(p.numel() for p in net.online_enc.parameters() if p.requires_grad)
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    print(f"[cfg] optimizer trains {n_opt} tensors | online-encoder trainable "
          f"tensors={n_enc} ({n_enc_params/1e6:.2f}M params)")
    if n_enc == 0:
        raise SystemExit(
            "HALT: online ENCODER has 0 trainable params → BYOL would train only the "
            "proj/pred heads and save a COCO backbone. Ensure requires_grad_(True) on "
            "net.online_enc (Ultralytics loads yolov8n.pt frozen).")
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and dev.type == "cuda"))

    start_ep, step = 0, 0
    state_path = WORK_ROOT / f"{OUT_NAME}_{src}_state.pt"
    if RESUME:
        start_ep, step = load_state(Path(RESUME), net, opt, scaler, dev)

    # DataParallel the WHOLE step (encoder+proj+pred+target) → both T4s share each
    # batch. EMA + optimizer act on `net` (the master); DP re-replicates each fwd.
    run = net
    if MULTI_GPU and torch.cuda.device_count() > 1:
        run = nn.DataParallel(net)
        print(f"[cfg] ⚠ DataParallel over {torch.cuda.device_count()} GPUs — WARNING: this path has "
              f"a PROVEN ~1 MB/step host-RAM LEAK (OOMs the 112k run at ~epoch 16) and ~no speedup "
              f"for yolov8n@batch{BATCH}. Strongly prefer MULTI_GPU=0 (two single-GPU jobs for 2×T4).")

    # MAX_STEPS (probe mode) must govern the SCHEDULES too, not just the stopping point.
    # Otherwise the LR/tau cosine is stretched over the full epoch budget and the probe never
    # traverses the LR band the real run died in. WARMUP_STEPS keeps the warmup:total ratio
    # faithful under compression (the 112k run was 5 warmup / 30 epochs = 1/6).
    total_steps = max(1, MAX_STEPS if MAX_STEPS else epochs * len(loader))
    warm = max(1, WARMUP_STEPS if WARMUP_STEPS else WARMUP_EP * len(loader))
    print(f"[train] {len(loader)} steps/epoch × {epochs} epochs "
          f"(start@{start_ep}) = {total_steps} steps")

    emb_std = float("nan")
    tau = TAU_BASE          # VICReg has no EMA target, so the loop never assigns tau —
                            # the epoch summary still prints it. (Crashed the A3 arm once.)
    # Collapse reference, measured on the UNTRAINED encoder at step 0 (below). A fixed
    # absolute threshold would be wrong: healthy per-dim std of an L2-normalised d-dim
    # embedding scales as 1/sqrt(d), so the bar has to be relative to where we started.
    ref_std, ref_enc_std, collapse_run, stop_all = 0.0, float("nan"), 0, False
    if OBJECTIVE == "vicreg" and MULTI_GPU:
        print("[cfg] ⚠ OBJECTIVE=vicreg with MULTI_GPU=1: VICReg's variance/covariance "
              "terms are BATCH statistics, so DataParallel computes them per-replica "
              "(per half-batch) — a different objective. Use MULTI_GPU=0.", flush=True)
    print(f"[cfg] objective={OBJECTIVE} target_bn={TARGET_BN_MODE} grad_clip={GRAD_CLIP} "
          f"collapse_halt={COLLAPSE_HALT} probe_log={PROBE_LOG} max_steps={MAX_STEPS}")
    print("[train] entering loop — waiting for the first batch "
          f"(WORKERS={WORKERS}; if this hangs, set WORKERS=0) …", flush=True)
    t_eta0 = time.time()          # reset on the 1st real step; see the ETA guard below
    for ep in range(start_ep, epochs):
        if stop_all:
            break
        net.set_modes()
        t0 = time.time()
        run_loss, run_std, nb = 0.0, 0.0, 0
        t_mark = time.time()
        for i, batch in enumerate(loader):
            t_load = time.time() - t_mark                # CPU: decode (+CPU aug if not GPU_AUG)
            hb = ep == start_ep and (i < 3 or (i + 1) % max(1, len(loader) // 10) == 0)
            if gpu_aug is not None:                      # kornia: augment the batch ON GPU
                x = batch.to(dev, non_blocking=True).float().div_(255.0)
                with torch.no_grad():
                    v1, v2 = gpu_aug[0](x), gpu_aug[1](x)
            else:
                v1, v2 = batch
                v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
            if ref_std == 0.0:                           # step 0: the "healthy" reference
                # The reference forward runs in TRAIN mode (so it measures the same thing the
                # loop does), which would nudge BN running stats and the target's copied
                # buffers. Snapshot + restore them so enabling these knobs cannot perturb a
                # real run — "defaults reproduce the shipped script" has to stay literally true.
                _bufs = [b.detach().clone() for b in net.buffers()]
                _r = probe_stats(net, v1)
                net.set_modes()
                with torch.no_grad():
                    _lv, _pn = run(v1, v2)
                    ref_std = max(1e-8, _pn.std(dim=0).mean().item())
                with torch.no_grad():
                    for _b, _s in zip(net.buffers(), _bufs):
                        _b.copy_(_s)
                del _bufs
                ref_enc_std = _r["z_enc_std"]
                print(f"[ref] untrained-encoder baseline: emb_std={ref_std:.5f} "
                      f"z_enc_std={ref_enc_std:.5f} bn_var_min={_r['bn_var_min']:.3e} "
                      f"→ collapse halt fires under {COLLAPSE_FLOOR * ref_std:.5f} "
                      f"for {COLLAPSE_PATIENCE} steps", flush=True)
                del _lv, _pn
            if step < warm:                              # linear warmup …
                lr = LR * (step + 1) / warm
            else:                                        # … then cosine decay to 0
                prog = (step - warm) / max(1, total_steps - warm)
                lr = 0.5 * LR * (1.0 + math.cos(math.pi * min(1.0, prog)))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss_vec, pn = run(v1, v2)                    # AMP is inside forward
            loss = loss_vec.mean()                        # DP gathers per-sample → mean
            lval = loss.item()
            if not math.isfinite(lval):                   # HALT 1: non-finite loss
                raise SystemExit(
                    f"HALT: non-finite loss ({lval}) at epoch {ep+1} step {step}. The 2026-07-27 "
                    f"112k run rode NaN for 20 epochs (~5.3h) before anyone saw it. Nothing is "
                    f"recoverable past this point — the last finite checkpoint is on disk.")
            scaler.scale(loss).backward()
            g_enc = float("nan")
            if GRAD_CLIP > 0 or (PROBE_LOG and step % PROBE_LOG == 0):
                scaler.unscale_(opt)                      # unscale once; clip and/or measure
                g_enc = float(torch.nn.utils.clip_grad_norm_(
                    online_params(net), GRAD_CLIP if GRAD_CLIP > 0 else float("inf")))
            scaler.step(opt); scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < 1e-4:   # HALT 2: scaler underflow
                raise SystemExit(
                    f"HALT: AMP GradScaler underflowed (scale={scaler.get_scale():g}) at epoch "
                    f"{ep+1} step {step} — it has been backing off on inf/NaN grads every step. "
                    f"Scale reaching 0 is a ONE-WAY DOOR: it makes every weight NaN (that is "
                    f"exactly what killed the 112k run). Fix the instability, don't retry.")
            if OBJECTIVE != "vicreg":                     # VICReg has no EMA target
                tau = cosine(step, total_steps, TAU_BASE, 1.0)
                update_target(net, tau)
            with torch.no_grad():
                p_std = pn.std(dim=0).mean().item()
                run_std += p_std
            # HALT 3: sustained collapse. Does NOT prevent collapse — it stops us paying
            # 8 hours to discover it. The reference is measured at step 0 (see below).
            if COLLAPSE_HALT and ref_std and p_std < COLLAPSE_FLOOR * ref_std:
                collapse_run += 1
                if collapse_run >= COLLAPSE_PATIENCE:
                    raise SystemExit(
                        f"HALT: representation COLLAPSE — emb_std {p_std:.5f} stayed under "
                        f"{COLLAPSE_FLOOR:.0%} of the step-0 reference {ref_std:.5f} for "
                        f"{collapse_run} consecutive steps (epoch {ep+1}, step {step}). "
                        f"The encoder is going constant; the saved backbone would be useless.")
            else:
                collapse_run = 0
            if PROBE_LOG and step % PROBE_LOG == 0:
                st = probe_stats(net, v1)
                net.set_modes()                           # probe_stats ran eval(); restore
                print(f"  [probe ep{ep+1} step{step}] loss={lval:.4f} "
                      f"z_enc_std={st['z_enc_std']:.5f} (ref {ref_enc_std:.5f}) "
                      f"bn_var_min={st['bn_var_min']:.3e} "
                      f"z_proj_std={st['z_proj_std']:.5f} p_std={p_std:.5f} "
                      f"g_enc={g_enc:.3f} scale={scaler.get_scale():g}", flush=True)
            run_loss += lval; nb += 1; step += 1
            if MAX_STEPS and step >= MAX_STEPS:
                print(f"[probe] MAX_STEPS={MAX_STEPS} reached — stopping.", flush=True)
                stop_all = True
                break
            if MEM_EVERY and step % MEM_EVERY == 0:       # leak isolation: intra-epoch RSS/sysRAM slope
                print(f"  [mem ep{ep+1} step{step}] {mem_report()}", flush=True)
            if hb:                                        # split load(CPU) vs compute(GPU)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_step = time.time() - t_mark - t_load
                print(f"  · batch {i+1}/{len(loader)}: load {t_load:5.2f}s  "
                      f"compute {t_step:5.2f}s  loss={loss.item():.3f}", flush=True)
            # ── ETA GUARD ──────────────────────────────────────────────────────
            # Print a projected wall-clock ONCE, early, so a config that cannot
            # finish is abortable in the first minute instead of at the 12h session
            # kill. ResNet-50 is ~18x the FLOPs of the yolov8n backbone, so the
            # yolov8n arm's SSL_IMGSZ=512 lands around 30h here — that mistake costs
            # a whole Kaggle session (and on a borrowed account, a favour).
            if step == 1:
                t_eta0 = time.time()      # skip step 0: cudnn autotune + warmup skew it
            if step == ETA_AFTER_STEPS and not RESUME:
                per_step = (time.time() - t_eta0) / max(1, ETA_AFTER_STEPS - 1)
                total_h = per_step * len(loader) * epochs / 3600
                print(f"\n[eta] {per_step*1000:.0f} ms/step · {per_step*len(loader)/60:.1f} min/epoch "
                      f"· {epochs} epochs → ~{total_h:.1f} h total", flush=True)
                if total_h > SESSION_HOURS:
                    print(f"[eta] ⚠⚠ THIS WILL NOT FINISH in a {SESSION_HOURS:.0f} h session. "
                          f"Abort now and lower SSL_IMGSZ / EPOCHS / N_IMAGES — or plan to "
                          f"RESUME from a state_ep*.pt in a second session (SAVE_EVERY is "
                          f"{SAVE_EVERY or 'OFF — set it!'}).", flush=True)
                else:
                    print(f"[eta] fits the {SESSION_HOURS:.0f} h session budget ✓\n", flush=True)
            t_mark = time.time()

        dt = time.time() - t0
        emb_std = run_std / nb
        flag = "  <-- WARNING: possible collapse" if emb_std < 1e-3 else ""
        # gc + empty_cache each epoch (cheap, ~10 min apart) — frees Python cycles +
        # GPU cache frag; with the printed mem it also tells us if a leak persists.
        import gc
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[ep {ep+1:03d}/{epochs}] loss={run_loss/nb:.4f}  "
              f"emb_std={emb_std:.4f}{flag}  lr={lr:.2e}  tau={tau:.4f}  {dt:.1f}s  "
              f"[{mem_report()}]")
        if TRACEMALLOC:                                   # name the biggest Python allocators (leak hunt)
            import tracemalloc
            top = tracemalloc.take_snapshot().statistics("lineno")[:8]
            print("  [tracemalloc top Python allocators]")
            for s in top:
                print(f"    {s}", flush=True)

        # Periodic: a resumable state.pt AND a loadable detector init, so a
        # 12h-capped session is neither all-or-nothing nor un-continuable.
        if SAVE_EVERY and (ep + 1) % SAVE_EVERY == 0 and (ep + 1) < epochs:
            # EPOCH-TAGGED, not a rolling `_latest.pt`. The 112k run overwrote one rolling
            # file every 5 epochs, so when it NaN'd at ep10 the good ep5 checkpoint had
            # already been destroyed by its NaN successors. Tagged files cost disk and
            # save the run. (`_latest.pt` is still written as a convenience symlink-ish copy.)
            save_state(WORK_ROOT / f"{OUT_NAME}_{src}_state_ep{ep+1:03d}.pt",
                       net, opt, scaler, ep + 1, step)
            save_state(state_path, net, opt, scaler, ep + 1, step)
            save_any(yolo, net, WORK_ROOT / f"{OUT_NAME}_{src}_ep{ep+1:03d}.pt",
                     meta={"method": OBJECTIVE.upper(), "init_source": src, "epoch": ep + 1,
                           "of": epochs, "emb_std": round(emb_std, 4), "note": "intermediate"},
                     dev=dev)

    # Objective is stamped into the filename — a VICReg init and a BYOL init are NOT
    # interchangeable and must never collide in weights/pretrain/.
    obj_tag = "" if OBJECTIVE == "byol" else f"{OBJECTIVE}_"
    # In probe mode EPOCHS is just a large cap that MAX_STEPS cuts short, so name the
    # artifact by STEPS — "ep10000" on a 3000-step probe would be a lie on disk.
    len_tag = f"s{MAX_STEPS}" if MAX_STEPS else f"ep{epochs}"
    tag = "smoke" if SMOKE else (f"{obj_tag}{src}_n{N_IMAGES or len(paths)}_{len_tag}")
    out_path = WORK_ROOT / f"{OUT_NAME}_{tag}.pt"
    save_any(yolo, net, out_path, meta={
        # ⚠ BASE_MODEL is the *yolov8n-arm* knob (env MODEL, default "yolov8n.pt") and is
        # meaningless on the resnet50 arm, which builds its trunk from torchvision. It was
        # recorded verbatim, so byol_resnet50_coco_n112120_ep18.pt carries
        # base_model="yolov8n.pt" — wrong provenance on a ResNet-50 file. Same class of bug
        # as exp11's hard-coded init_kind. Report per-arch instead.
        "method": OBJECTIVE.upper(), "init_source": src,
        "base_model": (f"torchvision retinanet_resnet50_fpn_v2 ({src})"
                       if ARCH == "resnet50" else BASE_MODEL),
        "corpus": "unlabelled CXR (NIH ChestX-ray14)",
        "n_images": len(paths), "ssl_imgsz": SSL_IMGSZ, "epochs": epochs,
        "batch": BATCH, "seed": SEED, "final_emb_std": round(emb_std, 4),
    })
    print(f"\nDONE. BYOL-adapted {ARCH} backbone → {out_path}")
    print("Download it, place at repo weights/pretrain/, then:")
    if ARCH == "resnet50":
        print(f"  # next rung: BYOL → VinDr (supervised detection on VinDr-CXR)")
        print(f"  BACKBONE_WEIGHTS=weights/pretrain/{out_path.name} \\")
        print(f"  python yolo_experiments/retinanet.py     # or the VinDr trainer once built")
    else:
        print(f"  MODEL=weights/pretrain/{out_path.name} \\")
        print("  python yolo_experiments/exp11_byol_init.py --aug-level mosaic_mixup")


if __name__ == "__main__":
    main()
