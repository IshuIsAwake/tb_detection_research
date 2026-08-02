"""
faster_rcnn.py — Fine-tune a COCO-pretrained torchvision Faster R-CNN (ResNet50-FPN
                 v2) on the frozen TB split, scored on the sealed 360 like the YOLO
                 champion (Active mAP50 0.745). Two-stage detector — the RPN +
                 ROI-head family — put next to the one-stage RetinaNet and YOLO to
                 see whether a proposal-based architecture attacks the recall wall
                 differently on the SAME data.

    All logic is shared in yolo_common/tv_detect.py (repo no-copy-paste rule);
    retinanet.py is the one-stage sibling. See that module's docstring for the
    comparability + framework caveats.

USAGE
    TB_SMOKE=1 python yolo_experiments/faster_rcnn.py              # plumbing (CPU ok)
    python yolo_experiments/faster_rcnn.py --epochs 40 --batch 2   # user runs (GPU)
    SEED=1 python yolo_experiments/faster_rcnn.py                  # another seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_common import tv_detect


def main() -> None:
    ap = argparse.ArgumentParser(description="Faster R-CNN (torchvision) on the frozen "
                                 "TB split, sealed-360 eval vs YOLO 0.745.")
    tv_detect.add_common_args(ap)
    tv_detect.run("fasterrcnn", ap.parse_args())


if __name__ == "__main__":
    main()
