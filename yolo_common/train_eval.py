"""
train_eval.py — Deterministic train + run-dir helpers (backbone-agnostic body).

Seeds Python/NumPy/torch and passes the seed to Ultralytics. AMP off, single
deterministic train. Returns the Ultralytics run dir + best.pt + train seconds.
The experiment script owns assembly of the final metrics.json envelope.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

from yolo_common import aug as AUG, settings as S


def seed_everything(seed: int = None) -> None:
    seed = S.SEED if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def train(data_yaml: Path, imgsz: int, project: Path, name: str, *,
          model: str = None, epochs: int = None, batch: int = None,
          device: str = None, aug_level: str = None, freeze: int = None,
          patience: int = None, lr0: float = None, workers: int = None) -> dict:
    """Train a YOLO model from pretrained weights. Any arg left None falls back
    to the (env-overridable) value in settings. TB_SMOKE forces 2 epochs.
    Returns a run-info dict including the resolved config."""
    from ultralytics import YOLO

    model = model or S.MODEL
    aug_level = aug_level or S.AUG_LEVEL
    epochs = S.SMOKE_EPOCHS if S.SMOKE else (epochs if epochs is not None else S.EPOCHS)
    batch = batch if batch is not None else S.BATCH
    device = device if device is not None else S.DEVICE
    freeze = freeze if freeze is not None else S.FREEZE
    patience = patience if patience is not None else S.PATIENCE
    lr0 = lr0 if lr0 is not None else S.LR0
    workers = workers if workers is not None else S.WORKERS

    seed_everything()
    yolo = YOLO(model)
    train_kwargs = dict(AUG.AUG_LEVELS[aug_level])
    if freeze is not None:
        train_kwargs["freeze"] = freeze
    if lr0 is not None:
        train_kwargs["lr0"] = lr0

    t0 = time.time()
    yolo.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        seed=S.SEED,
        deterministic=True,
        amp=S.AMP,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=True,
        **train_kwargs,
    )
    train_sec = round(time.time() - t0, 1)

    run_dir = Path(yolo.trainer.save_dir)
    return {
        "run_dir": run_dir,
        "best_pt": run_dir / "weights" / "best.pt",
        "train_sec": train_sec,
        "resolved": {"model": model, "epochs": epochs, "batch": batch,
                     "device": device, "aug_level": aug_level, "freeze": freeze,
                     "patience": patience, "lr0": lr0, "workers": workers},
    }


def load_best(best_pt: Path):
    from ultralytics import YOLO
    return YOLO(str(best_pt))
