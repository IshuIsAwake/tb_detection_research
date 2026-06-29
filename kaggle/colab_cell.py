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
# ──────────────────────────────────────────────────────────────────────────────

import os, subprocess, sys
from pathlib import Path

# 1. deps + Drive (the runtime dies if you close the tab; Drive is the safety net)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ultralytics"], check=True)
from google.colab import drive
drive.mount("/content/drive")

# 2. paths — point DRIVE at the folder where you uploaded the bundle (see README).
#    Bundle must contain: splits.json, tb/, labels_all/, vindr_pretrain_1024.pt,
#    tbx_train.py
DRIVE    = "/content/drive/MyDrive/tb"             # <-- adjust if you used another folder
BUNDLE   = DRIVE                                   # splits.json + tb/ + labels_all/ live here
VINDR    = f"{DRIVE}/vindr_pretrain_1024.pt"       # the 1024 VinDr backbone (init)
TRAIN_PY = f"{DRIVE}/tbx_train.py"                 # uploaded copy of kaggle/tbx_train.py
RUNS     = f"{DRIVE}/colab_runs"                    # checkpoints/best.pt — survive disconnects
Path(RUNS).mkdir(parents=True, exist_ok=True)

for p in (VINDR, TRAIN_PY, f"{BUNDLE}/splits.json"):
    assert Path(p).exists(), f"missing on Drive: {p}"

# 3. this account's two jobs (both augs, its seed). VinDr-1024 init for both.
JOBS = [("mosaic", SEED), ("mosaic_mixup", SEED)]

for aug, seed in JOBS:
    run_name = f"tbx_vindr1024_{aug}_1024_b16_s{seed}"
    run_dir  = Path(RUNS) / run_name
    if (run_dir / "DONE").exists():
        print(f"SKIP {run_name} — already DONE")
        continue
    resume = (run_dir / "weights" / "last.pt").exists()
    env = dict(
        os.environ,
        MODEL=VINDR, IMGSZ="1024", BATCH="16", AUG=aug, SEED=str(seed),
        EPOCHS="200", PATIENCE="100", DEVICE="0", WORKERS="2",
        INPUT_ROOT=BUNDLE,
        WORK_ROOT="/content/work",        # local/fast: materialised symlink tree
        PROJECT_ROOT=RUNS,                # Drive: checkpoints persist a disconnect
        RUN_NAME=run_name,
        RESUME="1" if resume else "0",
    )
    print(f"\n>>> {run_name}  (resume={resume})")
    subprocess.run([sys.executable, TRAIN_PY], env=env, check=True)

print("\nAll jobs for this account finished. Download from Drive:")
print(f"  {RUNS}/tbx_vindr1024_*_1024_b16_s{SEED}/weights/best.pt")
print("Then eval LOCALLY (do NOT eval on Colab):")
print("  python yolo_experiments/eval_weights.py --weights <best.pt> --imgsz 1024 "
      "--name <run_name>")
