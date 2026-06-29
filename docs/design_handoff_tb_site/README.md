# Handoff: TB Detection research site

A developer-ready package for turning the design-system + research-site prototype into a real, continuously-developed
website deployed to **GitHub Pages**. Pair this with `CLAUDE.md` (the task brief). This README is self-sufficient: a
developer who wasn't in the design conversation can implement the whole thing from here.

---

## 1. Overview

The site is a **research notebook** for a tuberculosis-detection-from-chest-X-rays project — a GitHub-hosted reference
page / working log that will be **updated continuously** as experiments land. It has four screens:

| Route | Screen | Purpose |
|---|---|---|
| `/` | **Home** | Paper-style landing: title, problem statement, abstract, background, aims, method. |
| `/results` | **Results & ablation** | A "Note" box (optimisation target + split policy), the latest run highlight, and a collapsible **YOLO Experiments** table where each run expands into Description / Metrics / Settings. |
| `/data` | **Data** | Survey of public TB / CXR datasets as expandable rows with capability marks, an Access column, role badges, stats and nuances. |
| `/references` | **References** | Literature reading list grouped by topic, each with a read-tracker + progress bar. |

A **light/dark toggle** lives in the top bar and persists.

The whole thing sits on a small in-repo **design system**: warm "Clinical Paper" CSS tokens + a set of React
components (Badge, Button, Tag, StatCard, ExperimentRow, Callout, PaperRef, Tabs). Keep that as a clean reusable
layer — it is the spine of every screen and of anything you add later.

---

## 2. About the design files

The files in `site/` and `design-system/` are **working React, but they run as a browser prototype**: JSX is
transpiled at load time by Babel, and modules find each other through `window.*` globals instead of imports. They are
the **authoritative design and behaviour** — your task is to **recreate them in a real build environment** (Vite +
React + TypeScript, recommended below), not to ship the prototype as-is. The look is **high-fidelity and final**:
match colours, type, spacing and interactions exactly; do not redesign.

`design-system/components/**/*.jsx` are already idiomatic React modules (`import React from "react"`,
`export function …`) with sibling `.d.ts` type signatures — these port over almost verbatim. Only the **site** files
(`Shell.jsx`, the `*Page.jsx`, `data.jsx`) carry the `window.*` prototype wiring that must be converted to imports.

---

## 3. Fidelity

**High-fidelity.** Final colours, typography, spacing, radii, shadows and interactions are all specified in the
token files and used literally by the components. Recreate pixel-for-pixel using the components as written.

---

## 4. Recommended target architecture

Vite + React 18 + TypeScript. Suggested tree:

```
tb_detection_research/
  index.html
  package.json
  tsconfig.json
  vite.config.ts                  # base: "/tb_detection_research/"
  .github/workflows/deploy.yml
  src/
    main.tsx                      # ReactDOM root + <HashRouter>
    App.tsx                       # layout shell + <Routes>
    lib/useTheme.ts               # data-theme + localStorage
    design-system/
      index.ts                    # barrel: re-export every component
      styles.css                  # @imports the tokens (unchanged)
      tokens/                     # colors / fonts / typography / spacing / radius-shadow (.css, unchanged)
      components/
        Badge.tsx Button.tsx Tag.tsx StatCard.tsx
        ExperimentRow.tsx Callout.tsx PaperRef.tsx Tabs.tsx
    components/
      Shell.tsx                   # TopBar, Wordmark, Icon, ThemeToggle, Page, PageHeader, SectionTitle, Eyebrow, NAV
    pages/
      OverviewPage.tsx ResultsPage.tsx DataPage.tsx ReferencesPage.tsx
    data/
      content.ts                  # EXPERIMENTS, NEXT_UP, DATASETS, PAPER_GROUPS, RUNS, RUN_DATE + helpers
```

`tsconfig.json` / `vite.config.ts`: set `@` → `src`. In `main.tsx`, `import "./design-system/styles.css";` once.

### Why HashRouter
GitHub **project** Pages serve from a subpath and don't rewrite unknown paths to `index.html`, so a normal
`BrowserRouter` 404s on deep links / refresh. `HashRouter` (`/#/results`) needs no server config and is the
lowest-friction choice. If you prefer clean URLs, use `BrowserRouter` with `basename="/tb_detection_research"` **and**
add a `public/404.html` that redirects into the SPA.

---

## 5. Migration map (prototype → modules)

| Prototype (now) | Target |
|---|---|
| `const { X } = window.TBDetectionResearchDesignSystem_ccf532` | `import { X } from "@/design-system"` |
| `window.OverviewPage = OverviewPage` (and the other pages) | `export function OverviewPage()` |
| `window.EXPERIMENTS / RUNS / DATASETS / PAPER_GROUPS / NEXT_UP / RUN_DATE` | `import { … } from "@/data/content"` |
| `window.buildRunMetrics / buildRunSettings / faTone` | `import { … } from "@/data/content"` |
| `window.Icon / Page / PageHeader / SectionTitle / Eyebrow / Wordmark / App` | named exports of `components/Shell.tsx` |
| `data.js.jsx` | `data/content.ts` (typed) |
| React/ReactDOM/Babel CDN `<script>` + `_ds_bundle.js` | delete — Vite compiles components directly |
| `App`'s `page` state + `setPage` | `react-router` routes + `<NavLink>`; `setPage("results")` → `navigate("/results")` |

Notes:
- `Icon` renders inline **Lucide** path data inside React (no DOM mutation) — keep it, or replace usages with
  `lucide-react` (`home`, `flask-conical`, `database`, `book-open`, `arrow-right`, `arrow-up-right`,
  `circle-dashed`, `check`, `minus`, `x`, `git-branch`, `sun`, `moon`).
- A few spots use `dangerouslySetInnerHTML` for short trusted strings containing `<strong>` / `<s>` (findings,
  next-up). Content is static and authored in-repo — safe to keep, or convert to JSX.
- `ResultsPage` no longer uses `ExperimentRow`; keep the component in the design system anyway (it's a system
  primitive and is exercised by the design-system showcase).

---

## 6. Screens in detail

### Home (`OverviewPage.tsx`) — centered, max-width 820px
- **Title block:** `<h1>` "Detecting tuberculosis in chest X-rays" (serif, `clamp(2.4rem,5vw,3.4rem)`, weight 600,
  `--ls-tight`, line-height 1.08), centered. Below it a centered lead paragraph (the problem statement, `--text-lead`,
  `--ink-2`, max ~60ch) and a centered button row (`primary` "Read results & ablation" with right arrow, `secondary`
  "Dataset research", `ghost` "References"). A 120px hairline divider follows.
- **Paper sections** via a `PaperSection` helper (mono eyebrow label in `--primary` + serif `<h2>` + centered prose,
  max ~62ch): **Abstract**, **01 · Background**, **02 · Aims** (two `GoalCard`s in a 2-col grid — `01` Active /
  `02` Planned), **03 · Method** (prose + a centered row of `Tag`s). All prose centered.

### Results & ablation (`ResultsPage.tsx`)
- **Header:** serif `<h1>` "Results & ablation" only — no eyebrow, no lead.
- **Note box:** a `Callout kind="note" title="Note"` containing two lines:
  - **a.** "We optimize mAP and lesion-level recall while balancing localization."
  - **b.** "On non-CV runs we use a frozen split. On CV runs the test set is always disjoint. (CV = cross-validation)"
- **Latest run:** highlight card for `RUNS[0]` — mono primary title, "Latest run" + "augmentation finalist" badges,
  a sentence with a `mosaic @ 512, batch 16` tag, and a `StatCard` row (Active mAP50, Recall, TB caught @.25,
  Healthy FA @.25, Matched IoU) with semantic tones.
- **Experiments → YOLO Experiments:** a collapsible panel (`--r-lg` card, header bar reading "YOLO Experiments ·
  N runs · newest first" with a chevron). Inside, a CSS-grid table.
  - **Grid template (header + every row):** `1.7fr 0.7fr 0.9fr 0.72fr 0.95fr 1.05fr 0.85fr 0.8fr 34px`.
  - **Columns:** Experiment · mAP50 · Precision · Recall · TB caught @.25 · **FA @.25 (H/S)** · Train time · Date · chevron.
  - **FA column is combined**: healthy and sick rendered together as `H% / S%`, each coloured by `faTone`.
  - **Date** column shows `RUN_DATE` ("27 Jul 2026") for all rows.
  - Numbers are mono, tabular. mAP50 and Recall get a green tone when strong. First row carries a "best" badge.
  - **Expanding a row** reveals a `Tabs` segmented control → **Description / Metrics / Settings**:
    - *Description*: the run's prose in a `--paper-2` well.
    - *Metrics*: colour-coded `StatCard` boxes in two columns — **Detection** (mAP50 overall, mAP50-95, Precision,
      Recall), **Active TB**, **Obsolete TB**; **Localization** (IoU), **Screening** (a `@0.10/@0.25/@0.50` `Tabs`
      toggle over TB detect rate / TB flagged / Healthy FA / Sick FA), **Recall by lesion size** (Small / Medium /
      Large). See §8 for how these are computed.
    - *Settings*: training-config `StatCard`s (Model, Image size, Batch, Epochs, Patience, Augmentation, Optimizer,
      lr0, Seed, Sampling).
- **Next (Planned):** `NEXT_UP` items as `Tag` + text rows.

### Data (`DataPage.tsx`)
- `PageHeader` (eyebrow "Home / Data", title "Dataset research", lead). Survey of `DATASETS` as expandable rows.
- **Row grid template (header + rows):** `1.5fr 0.95fr 0.5fr 0.5fr 0.7fr 0.8fr auto`.
- **Columns:** Dataset · TB images · BBox · Seg · **Access** · Role · chevron. (The Access column is now a top-level
  column — a `Badge tone="primary"` with the access value — and was removed from the expanded body to avoid
  duplication.) `CapMark` renders ✓ / ~ / ✗ for BBox & Seg. Expanded body: a Statistics `StatCard` grid + an
  Annotations tag list (left) and Key-nuances list + note + "Open dataset" link (right).

### References (`ReferencesPage.tsx`)
- Reading list grouped by topic; each paper is a `PaperRef` with a working read-tracker and a per-group progress bar.
  (Unchanged from the original; port verbatim.)

---

## 7. Top bar, theme & nav (`Shell.tsx`)
- Sticky translucent header (`color-mix(in srgb, var(--paper) 82%, transparent)` + blur), `--content-max` 1180px.
- **Wordmark** (TB monogram tile + serif "TB Detection") → links Home.
- **NAV** = `[{Home,/}, {Results & ablation,/results}, {Data,/data}, {References,/references}]`, each a `.nav-link`
  with `.active` styling (the small CSS block lives in the prototype `index.html` — move it into `styles.css` or a
  component). Active state by route.
- A GitHub repo link, then the **ThemeToggle** (pill button, sun in dark / moon in light).
- **`useTheme`**: state seeded from `localStorage["tb-theme"]` (default `"light"`); effect sets
  `document.documentElement.dataset.theme = theme` and writes back to `localStorage`. Toggle flips light/dark.

---

## 8. Data model (`data/content.ts`)

Port `site/data.jsx` verbatim into a typed module. Key exports:

- **`RUNS`** — array of YOLO runs, newest first. Each: `name, group, aug, imgsz, batch, seed, map50, precision,
  recall, tbCaught ("108/121"), healthyFA ("2.5%"), sickFA, trainTime, date, desc`, optional `best`, optional
  `metrics` override.
- **`RUN_DATE`** = `"27 Jul 2026"` (placeholder applied to every row — replace with real dates when available).
- **`buildRunMetrics(run)`** — returns the full metrics object for the Metrics tab. For
  `exp4_posonly_mosaic_1024_s2` the deep figures (Active/Obsolete mAP, IoU, lesion-size recall) are **real**; for
  every other run they are **deterministically synthesised** from that row's headline numbers (active ≈ overall×1.61,
  obsolete ≈ overall×0.385, IoU from overall, lesion recall from recall; screening @0.10/@0.50 derived from @0.25).
  **These synthesised numbers are placeholders** — swap in real per-run metrics as they become available.
- **`buildRunSettings(run)`** — derives the Settings rows from the run fields.
- **`faTone(value, isHealthy)`** — maps a false-alarm % to a `good/warn/bad` tone.
- **`EXPERIMENTS`** (exp1–exp4 narrative, retained), **`NEXT_UP`**, **`DATASETS`**, **`PAPER_GROUPS`**.

⚠️ **Flag for the project owner:** only the per-run *table* values and the one fully-specified run's deep metrics are
real. The other runs' deep metrics and all `RUN_DATE`s are synthesised placeholders. Decide whether to wire these to
real outputs (e.g. parse the YOLO `results.csv` / a generated JSON) before publishing.

---

## 9. Design tokens (full)

All components are styled **only** through these CSS variables — ship `tokens/*.css` + `styles.css` unchanged.

**Light colours:** `--paper #FBFAF6` · `--paper-2 #F4F2EC` · `--surface #FFFFFF` · `--surface-2 #FCFBF8` ·
`--ink #1E2022` · `--ink-2 #4A4E52` · `--ink-3 #797E83` · `--ink-4 #A7ABAE` · `--line #E7E2D8` · `--line-2 #D6CFC1` ·
`--line-3 #BFB8A8` · `--primary #1C6B6E` · `--primary-strong #135053` · `--primary-tint #E2EEED` ·
`--primary-tint-2 #C9E0DE` · `--on-primary #FBFAF6` · `--good #2E7D5B` · `--warn #B5742A` · `--bad #B0473C` ·
`--info #2C6285` (+ matching `*-tint`s).

**Dark colours** (`[data-theme="dark"]`): `--paper #15171A` · `--paper-2 #1D2024` · `--surface #1E2226` ·
`--surface-2 #24282D` · `--ink #ECEAE3` · `--ink-2 #BFBDB4` · `--ink-3 #8A8F94` · `--ink-4 #5C6266` ·
`--line #2E333A` · `--line-2 #3A4047` · `--line-3 #4B525A` · `--primary #4FBAB6` · `--primary-strong #6FCDC9` ·
`--primary-tint #15302F` · `--primary-tint-2 #214E4D` · `--on-primary #0C1413` · `--good #5FB98C` · `--warn #D49B5A` ·
`--bad #D9756A` · `--info #6BA8CE` (+ tints).

**Type:** families `--font-serif` IBM Plex Serif, `--font-sans` IBM Plex Sans, `--font-mono` IBM Plex Mono (Google
Fonts `@import` in `fonts.css`; base `html { font-size:17px }`). Scale: `--text-display 2.6rem`, `--text-h1 2rem`,
`--text-h2 1.5rem`, `--text-h3 1.2rem`, `--text-lead 1.18rem`, `--text-body 1rem`, `--text-sm 0.88rem`,
`--text-xs 0.78rem`, `--text-label 0.7rem`. Weights 400/500/600/700. Line-heights `--lh-tight 1.18`, `--lh-snug 1.35`,
`--lh-body 1.7`, `--lh-relaxed 1.85`. Letter-spacing `--ls-tight -0.02em`, `--ls-label 0.08em`.

**Spacing (4px base):** `--sp-1…9` = 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96px. Layout: `--content-max 1180px`,
`--prose-max 720px`, `--gutter 2rem`.

**Radius:** `--r-sm 4px`, `--r 8px`, `--r-lg 14px`, `--r-pill 999px`.
**Shadow:** `--shadow-sm`, `--shadow`, `--shadow-lg` (warm-tinted, low — see `radius-shadow.css`).
**Motion:** `--ease cubic-bezier(.32,.72,0,1)`, `--dur 200ms`, `--transition var(--dur) var(--ease)`.

---

## 10. GitHub Pages deploy

`vite.config.ts`:
```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  base: "/tb_detection_research/",                 // <-- repo name
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
});
```

`.github/workflows/deploy.yml`:
```yaml
name: Deploy to Pages
on:
  push: { branches: [main] }
  workflow_dispatch:
permissions: { contents: read, pages: write, id-token: write }
concurrency: { group: pages, cancel-in-progress: true }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with: { path: dist }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: { name: github-pages, url: "${{ steps.deploy.outputs.page_url }}" }
    steps:
      - id: deploy
        uses: actions/deploy-pages@v4
```

Then: repo **Settings → Pages → Source: GitHub Actions**. Push to `main` → live at
`https://<user>.github.io/tb_detection_research/`. (With `HashRouter`, deep links and refresh work with no extra config.)

---

## 11. Files in this bundle

```
design-system/
  styles.css                      # entry — @imports the tokens
  tokens/                         # colors.css (incl. dark block), fonts.css, typography.css, spacing.css, radius-shadow.css
  components/                     # Badge, Button, Tag, StatCard, ExperimentRow, Callout, PaperRef, Tabs
                                  #   each: .jsx (source) + .d.ts (types) + .prompt.md (intent) + a *.card.html showcase
site/
  index.html                     # prototype host (CDN React/Babel + the .nav-link CSS to migrate)
  Shell.jsx                      # chrome: TopBar, Wordmark, Icon, ThemeToggle, Page, PageHeader, SectionTitle, Eyebrow, App, NAV
  OverviewPage.jsx               # Home
  ResultsPage.jsx                # Results & ablation (Note box, latest run, YOLO Experiments table)
  DataPage.jsx                   # Data (with Access column)
  ReferencesPage.jsx             # References
  data.jsx                       # all content + RUNS + buildRunMetrics/buildRunSettings/faTone  → becomes data/content.ts
CLAUDE.md                        # the task brief / guardrails
README.md                        # this file
```

The `.d.ts` files are authoritative prop signatures — use them when typing the `.tsx` components. The `.prompt.md`
files explain each component's intent if you need it; the `*.card.html` files are isolated showcases (they load the
old global bundle — reference only, don't port).
