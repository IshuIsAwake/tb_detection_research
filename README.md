# TB detection on chest X-rays

Trying to improve tuberculosis detection. Right now I'm just seeing what works
and what doesn't.

Data: [TBX11K](https://github.com/yun-liu/Tuberculosis) (CC BY 4.0). Not in this
repo — download it and put it at `data/TBX11K/`.

## Layout

- `yolo_common/` — shared code (data prep, splits, training, metrics).
- `yolo_experiments/` — one script per experiment, results saved per run.
- `RESULTS.md` — what each run found, in order.
- `docs/` — the project page.

## Run an experiment

```bash
python yolo_experiments/exp1_yolo_baseline.py --imgsz 512        # plain baseline
python yolo_experiments/exp2_negatives.py     --imgsz 512 --batch 16   # + negatives
```

Most settings are env vars or flags (epochs, batch, model, etc.) — see the
script docstrings.

## Status

Detection baseline only so far (YOLOv8n). See `RESULTS.md` for current findings.

## Future ideas

Noted but not built — likely candidates for a later custom architecture, not the
current YOLO baseline:

- **Anatomy-preserving mosaic.** A quadrant-constrained CutMix that only places a
  region where it anatomically belongs (a top-left lung stays top-left), so the
  composite is still a plausible chest. Plain YOLO mosaic just juxtaposes four
  partial chests with no anatomy constraint.
- **Lesion-size-conditional blur.** Blur only images whose lesions are large
  enough to survive it — acquisition-sharpness robustness without erasing the
  small lesions we're trying to find.
