"""
retinanet.py — Fine-tune a COCO-pretrained torchvision RetinaNet (ResNet50-FPN v2)
               on the frozen TB split, scored on the sealed 360 like the YOLO
               champion (Active mAP50 0.745). One-stage detector; RetinaNet+P2T
               was SymFormer's own leaderboard baseline, so this is the most
               literature-aligned architecture to put next to YOLO.

    All logic is shared in yolo_common/tv_detect.py (repo no-copy-paste rule);
    faster_rcnn.py is the two-stage sibling. See that module's docstring for the
    comparability + framework caveats (big model on 559 images; AP is a
    self-contained COCO-style AP@0.5; mosaic/mixup are ours, not torchvision's).

USAGE
    TB_SMOKE=1 python yolo_experiments/retinanet.py --name smoke    # plumbing (CPU ok)
    python yolo_experiments/retinanet.py --epochs 40 --batch 2      # user runs (GPU)
    SEED=1 python yolo_experiments/retinanet.py                     # another seed

⚠ ALWAYS PASS --name ON A SMOKE RUN. The default tag is f"{arch}_s{SEED}" —
    i.e. `retinanet_s0`, a REAL recorded run (0.7393). Its results directory has
    already been destroyed once by a 2-epoch smoke run; metrics.json, summary.txt
    and best.pt are all overwritten in place, with no confirmation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import tv_detect


def main() -> None:
    ap = argparse.ArgumentParser(description="RetinaNet (torchvision) on the frozen "
                                 "TB split, sealed-360 eval vs YOLO 0.745.")
    tv_detect.add_common_args(ap)
    tv_detect.run("retinanet", ap.parse_args())


if __name__ == "__main__":
    main()
