"""yolo_common — shared code for the TB YOLO experiment harness.

Code only. Every generated artifact (converted labels, splits.json, data.yaml,
materialised dataset trees, per-run results) lives outside this package under
`yolo_datasets/` (shared, rebuildable) or `yolo_experiments/results/` (per run).

Experiment scripts import this package via a path shim (see any exp*.py):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from yolo_common import settings, convert, splits, metrics, train_eval
"""
