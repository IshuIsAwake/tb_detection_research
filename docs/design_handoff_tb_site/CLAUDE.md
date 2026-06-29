# CLAUDE.md — Task brief

You are setting up a **real, continuously-developed website** from a design handoff. Read `README.md` in this
folder for the full spec; this file is the actionable plan and the guardrails.

## What you're building
A static React site — the **TB Detection research notebook** — deployed to **GitHub Pages** at
`https://<user>.github.io/tb_detection_research/`. It is built on a small in-repo **design system** (CSS tokens +
React components) that must remain a clean, reusable layer because both the site and future surfaces depend on it.

## Source of truth
Everything in this handoff is the **intended design, already implemented as working React**. Your job is **not to
redesign** — it is to lift this code into a proper Vite + React + TypeScript project with real ES imports, real
routing, and a build/deploy pipeline. Match the look pixel-for-pixel; the prototypes are high-fidelity and final.

## The one real refactor
The prototype runs JSX in the browser via Babel and wires modules together through **`window.*` globals**. That is
the only thing that must change structurally. Concretely:

- `const { Button } = window.TBDetectionResearchDesignSystem_ccf532;` → `import { Button } from "@/design-system";`
- `window.OverviewPage = OverviewPage;` → `export function OverviewPage() {…}`
- `window.EXPERIMENTS`, `window.RUNS`, `window.buildRunMetrics`, … → `import { … } from "@/data/content";`
- `window.Icon`, `window.Page`, `window.SectionTitle`, `window.Eyebrow` (all defined in `Shell.jsx`) → import from the Shell module.
- Delete the React / ReactDOM / Babel `<script>` CDN tags and the `_ds_bundle.js` dependency — Vite compiles the components directly, so the bundle is no longer needed.

The design-system component files (`design-system/components/**/*.jsx`) are **already idiomatic React** (`import React`,
`export function`) — they move over almost verbatim; just rename to `.tsx` and add prop types from the sibling `.d.ts`.

## Do not change
- Token names or values (`--paper`, `--primary`, `--text-h2`, `--sp-5`, `--r-lg`, `--shadow`, …). The components are
  written entirely against these CSS variables — keep `styles.css` and `tokens/` as the styling layer.
- The dark-mode mechanism: `data-theme="dark"` on `<html>` + the `[data-theme="dark"]` block in `tokens/colors.css`,
  persisted to `localStorage["tb-theme"]`. Keep it; just wrap it in a `useTheme` hook.
- Component APIs, copy, layout, and the warm "Clinical Paper" aesthetic.

## Recommended target (see README for the full tree + configs)
- **Vite + React 18 + TypeScript**, `@` path alias → `src`.
- **Routing:** `react-router-dom` with **`HashRouter`** (zero-config on project Pages — no 404 rewrite needed). Map
  the four nav entries (Home, Results & ablation, Data, References) to routes; keep the active-state styling.
- **Icons:** keep the inline `Icon` component as-is (zero-dep), or swap to `lucide-react` (the path data is Lucide).
- **Deploy:** GitHub Actions → Pages, with `base: "/tb_detection_research/"` in `vite.config.ts`. Workflow YAML is in README.

## Definition of done
1. `npm run dev` renders all four pages identically to the prototype, light and dark.
2. URLs are shareable (hash routes), nav active-states correct, theme persists across reloads.
3. `npm run build` is clean; the Action deploys to Pages and the live site matches.
4. The design system is importable as one module (`@/design-system`) and still styled purely by tokens.

Start by reading `README.md`, then scaffold the project, port `tokens/` + `styles.css` + `components/`, then the data
module, then Shell + pages, then routing, then the deploy workflow.
