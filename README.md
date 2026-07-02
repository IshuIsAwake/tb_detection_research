# TB detection on chest X-rays

Refer to the website (github page).

Trying to improve tuberculosis detection. Right now I'm just seeing what works
and what doesn't.

Data: [TBX11K](https://github.com/yun-liu/Tuberculosis) (CC BY 4.0). Not in this
repo — download it and put it at `data/TBX11K/`.

## Layout

- `yolo_common/` — shared code (data prep, splits, training, metrics).
- `yolo_experiments/` — one script per experiment, results saved per run.
- `RESULTS.md` — what each run found, in order.
- `ideas.md` — parked ideas / future directions.
- `docs/` — the project page.

## Run an experiment

```bash
python yolo_experiments/exp1_yolo_baseline.py --imgsz 512        # plain baseline
python yolo_experiments/exp2_negatives.py     --imgsz 512 --batch 16   # + negatives
```

Most settings are env vars or flags (epochs, batch, model, etc.) — see the
script docstrings.

## Status

YOLO detection baseline is **locked**: VinDr-init + mosaic_mixup + full fine-tune
@ 512 = Active mAP50 0.745 on the sealed test. Model size, resolution, and
inference tricks are all exhausted — the wall is data (799 TB images), not the
model. Next: the two-stage **classifier → detector** pipeline. See `RESULTS.md`.

## Future ideas

Parked ideas and future directions live in [`ideas.md`](ideas.md) — augmentation,
preprocessing, and pipeline-topology (how to stage detectors and classifiers).
