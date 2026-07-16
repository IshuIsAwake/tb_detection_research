# CLAUDE.md

Project guidance for Claude Code and other AI assistants working in this repo.

## What this is

TB detection on chest X-rays (TBX11K). Long-term plan is a two-stage pipeline
(image-level classifier → TB detector); current work is the **detection baseline
only**. `data_research_report.md` is the authoritative dataset survey — read it
before touching data. `exp1_claude_code_prompt.md` is the original brief for
exp1.

## Layout (decided structure)

```
data/TBX11K/        read-only original data (COCO JSON + VOC XML). NEVER written.
yolo_common/        shared CODE only (settings, convert, splits, metrics,
                    train_eval, aug). Imported by experiments via a path shim.
yolo_datasets/      GENERATED, shared, rebuildable: converted labels,
                    splits.json, data.yaml, materialised trees. Safe to delete.
yolo_experiments/   thin per-experiment scripts + per-run results/.
```

Import mechanism (option 2A — path shim, no install):

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from yolo_common import settings, splits, metrics, train_eval
```

## Invariants you must not violate

- **`data/TBX11K/` is read-only.** Everything generated goes under
  `yolo_datasets/` (shared) or `yolo_experiments/results/` (per run). All
  generated paths are safe to delete and rebuild.
- **`splits.json` is frozen.** Built once by `yolo_common/splits.py`, never
  regenerated (the on-disk file is the authority). It is the reference split for
  EVERY experiment. Atomic write, deterministic from `SEED`.
- **Leakage rule:** `blackbox_negative_ids` (120 sick + 120 healthy in the
  sealed test set) are RESERVED. Any future experiment adding negatives to
  TRAINING must exclude these 240 (draw from the other ~3680 sick / ~3680
  healthy). The rule is recorded inside `splits.json` too.
- **exp1 is a deliberate FLOOR:** positives-only training, augmentation OFF,
  AMP OFF, no CV, no tuning. Do not "help" it. Its job is to expose failures.

## Landmines (each one costs a debugging session)

- **COCO JSON is the box coord source, NOT the XML.** `annotations/json/
  TBX11K_trainval_only_tb.json` uses 512×512 coords (matches the images). The
  VOC `annotations/xml/` uses ORIGINAL clinical resolution (~2840px) — feeding
  XML coords to a 512px image silently produces garbage boxes.
- **800 tb images, 799 carry boxes.** One tb image has no annotation and is NOT
  a detection positive. `EXPECT_POSITIVES = 799` — this matches the report's
  "~799 TB-positive", it is not a bug.
- **Class map: `0 = ActiveTuberculosis`, `1 = ObsoletePulmonaryTuberculosis`.**
  COCO ids are 1/2 (and a never-used id 3 `PulmonaryTuberculosis`). `convert.py`
  re-verifies the names against the JSON and halts on mismatch.
- **The "prompt says already-YOLO" trap.** `exp1_claude_code_prompt.md` Step 0
  claims the data is already in YOLO format. It is NOT — it ships as COCO/VOC.
  `yolo_common/convert.py` is the prep step that produces YOLO labels. Folder
  names are `health`/`sick`/`tb`/`test`/`extra` (not Healthy/Sick-non-TB/...).
- **Ultralytics label-path inference** swaps the `images` path segment for
  `labels` and the ext for `.txt`. The materialised tree
  (`yolo_datasets/det/{images,labels}/{train,val}`) uses exactly those names so
  symlinked images resolve to their labels. Don't rename those dirs.
- **AMP is off by design** (`settings.AMP = False`) — Ultralytics' AMP
  self-check downloads a helper model that 404s in some envs, and AMP is a
  confound for a deterministic floor.

## The GPU rule

The user runs **all** GPU work themselves — training **and** eval/inference —
and monitors it. Write the code and hand back the exact copy-pasteable command
with the required settings filled in; do **not** launch a real (non-smoke) GPU
run yourself. This covers local eval / inference passes too (e.g. the exp9
cross-domain sensitivity run), not only training. The only exception is
`TB_SMOKE=1` plumbing checks.

## Code style

Many small, single-responsibility files. Shared logic lives in `yolo_common/`
and is imported, never copy-pasted into experiment scripts (this is the lesson
from the Oral_cancer harness, which started single-file and refactored to a
shared package). Pure geometry/conversion helpers stay free of training deps.

## When docs and code disagree

Docs describe intent. Investigate the gap; do not silently "fix" code to match
docs or docs to match code. If the gap is real, surface it. The exp1 inventory
step (`inventory_or_halt`) enforces this for the data.

## The site (`site/`)

Vite + React + react-router on **HashRouter**, deployed to GitHub Pages. No UI,
chart or icon libraries — everything is inline SVG + CSS custom properties.

Layout: a collapsible left sidebar (`components/Sidebar.tsx`); there is no top
bar. `data/nav.ts` lists PAGES only — the sub-sections under the active page are
**discovered from the DOM** at runtime (any element with `data-section`, emitted
by `ReadSection`). So the prose is the single source of truth for the outline and
the two cannot drift. Don't add a parallel list of sections.

Rules that cost a debugging session if broken:

- **HashRouter means `href="#some-id"` is a ROUTE, not an anchor.** In-page jumps
  must be programmatic (`scrollToSection`).
- **Chart colours come from `tokens/viz.css`,** a teal-led palette derived from
  the brand accent and validated (all six checks pass, both modes, against this
  site's real surfaces). Slot order is the colour-blindness safety mechanism, not
  taste. Assign slots 1..6 in order, never cycle to a 7th. Nominal one-series
  bars all wear `--viz-1`; ordered stages use the `--viz-ramp-*` ramp. Re-run the
  dataviz validator if you touch a value.
- **Citations are per-section and deduped.** Each page owns its `SOURCES` and its
  superscripts restart at 1. The bibliography is a native `<details>` so a
  `<Cite>` can force it open by id. `makeCite` warns on duplicate URLs in dev.
- **Diagram hover handlers must use the DOM names** (`onMouseEnter`/`onMouseLeave`).
  The `on(part)` helper is spread onto raw `<g>` elements as well as `<Block>`;
  custom prop names silently no-op on a `<g>` (React only warns).
- **X-ray figures ship boxes as DATA, not burned into pixels** (`data/xrays.ts`,
  regenerate with `python site/scripts/build_xray_assets.py`). Boxes come from the
  COCO JSON at 512×512 — never the VOC XML. That's what makes the reveal toggle
  possible, and it keeps the images ~15 KB each.
- **Nothing may scroll the page horizontally.** Wide tables get their own
  `overflow-x: auto` container; long unbreakable run names need
  `overflow-wrap: anywhere`; grid tracks that hold them need `minmax(0, 1fr)`,
  since a bare `1fr` has a min-content floor that blows the grid out.

Verify with a real browser before calling site work done — `npm run build`, then
`npx vite preview`, then check every page at 390/768/1440 in both themes. Type
checking will not catch a dead hover handler or a zero-height hit target.
