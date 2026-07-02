# ──────────────────────────────────────────────────────────────────────────────
# Colab cell — TBX11K 1024@16 training, VinDr-1024 init, one account.
#
# WHY Colab (not Kaggle) for this batch: it's a morning job, internet is stable,
# and free Colab gives ONE GPU runtime per account — so 3 accounts run 3 seeds in
# parallel. Each account runs BOTH augs for its seed (2 jobs back-to-back, ~5 h).
# Kaggle is saved for overnight detached runs.
#
# NO background execution on free Colab: keep the tab open. Checkpoints go to
# Drive every epoch, so a disconnect costs only the in-flight epoch — re-run this
# same cell and it RESUMES from last.pt (or SKIPs a finished job).
#
# Eval stays LOCAL: download each best.pt and score with
# yolo_experiments/eval_weights.py (byte-identical to exp1-6). Do NOT eval here.
#
# ── per-account setup ─────────────────────────────────────────────────────────
# Account A → SEED = 0   |   Account B → SEED = 1   |   Account C → SEED = 2
SEED = 0

# ── WHAT TO TRAIN (change these for a new batch; everything else stays) ─────────
# MODEL: a bare Ultralytics name (auto-downloaded, needs internet) OR a Drive path.
#   yolov8s capacity probe  → "yolov8s.pt"  (COCO init; the VinDr backbone is
#                              yolov8n and CANNOT init an s model — arch mismatch).
#   VinDr-1024 batch        → f"{DRIVE}/vindr_pretrain_1024.pt"
MODEL = "yolov8s.pt"
INIT  = "yolov8s"            # short tag baked into run names (e.g. yolov8s | vindr1024)
IMGSZ = 1024
AUGS  = ["mosaic"]          # mosaic won at 1024; use ["mosaic", "mosaic_mixup"] for both
# ──────────────────────────────────────────────────────────────────────────────

import os, subprocess, sys
from pathlib import Path

# 1. deps + Drive (the runtime dies if you close the tab; Drive is the safety net)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ultralytics"], check=True)
from google.colab import drive
drive.mount("/content/drive")

# 2. paths — point DRIVE at the folder where you uploaded the bundle (see README).
#    Bundle must contain: splits.json, tb/, labels_all/, tbx_train.py (+ any Drive
#    init weights you reference in MODEL above).
DRIVE    = "/content/drive/MyDrive/tb"             # <-- adjust if you used another folder
BUNDLE   = DRIVE                                   # splits.json + tb/ + labels_all/ live here
TRAIN_PY = f"{DRIVE}/tbx_train.py"                 # uploaded copy of kaggle/tbx_train.py
RUNS     = f"{DRIVE}/colab_runs"                    # checkpoints/best.pt — survive disconnects
Path(RUNS).mkdir(parents=True, exist_ok=True)

# A Drive-path MODEL must exist; a bare "yolov8*.pt" name is downloaded, so skip it.
checks = [TRAIN_PY, f"{BUNDLE}/splits.json"]
if "/" in MODEL:
    checks.append(MODEL)
for p in checks:
    assert Path(p).exists(), f"missing on Drive: {p}"

# 3. this account's jobs (one per aug, its seed).
JOBS = [(aug, SEED) for aug in AUGS]

for aug, seed in JOBS:
    run_name = f"tbx_{INIT}_{aug}_{IMGSZ}_b16_s{seed}"
    run_dir  = Path(RUNS) / run_name
    if (run_dir / "DONE").exists():
        print(f"SKIP {run_name} — already DONE")
        continue
    resume = (run_dir / "weights" / "last.pt").exists()
    env = dict(
        os.environ,
        MODEL=MODEL, IMGSZ=str(IMGSZ), BATCH="16", AUG=aug, SEED=str(seed),
        EPOCHS="200", PATIENCE="100", DEVICE="0", WORKERS="2",
        INPUT_ROOT=BUNDLE,
        WORK_ROOT="/content/work",        # local/fast: materialised symlink tree
        PROJECT_ROOT=RUNS,                # Drive: checkpoints persist a disconnect
        RUN_NAME=run_name,
        RESUME="1" if resume else "0",
    )
    print(f"\n>>> {run_name}  (model={MODEL}, resume={resume})")
    subprocess.run([sys.executable, TRAIN_PY], env=env, check=True)

print("\nAll jobs for this account finished. Download from Drive:")
print(f"  {RUNS}/tbx_{INIT}_*_{IMGSZ}_b16_s{SEED}/weights/best.pt")
print("Then eval LOCALLY (do NOT eval on Colab):")
print(f"  python yolo_experiments/eval_weights.py --weights <best.pt> --imgsz {IMGSZ} "
      "--name <run_name>")
