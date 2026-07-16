import React from "react";
import { DiagramFrame } from "./primitives";

// A 6x6 patch grid standing in for a chest X-ray. Two lesions are planted in
// opposite corners — an apical infiltrate and a distant basal effusion — because
// the point of this figure is that a Transformer can relate them in one step
// while a CNN's early receptive field cannot.
const GRID = 6;
const APEX = [7, 8, 13]; // upper-right of the patient = image-left region
const BASE = [27, 28, 33];

const isLesion = (i: number) => APEX.includes(i) || BASE.includes(i);

/** Attention that any lesion patch pays to any other patch. */
function attentionFrom(src: number): number[] {
  return Array.from({ length: GRID * GRID }, (_, dst) => {
    if (dst === src) return 1;
    if (!isLesion(src)) {
      // A healthy patch attends locally — nothing interesting to look at.
      const [sr, sc] = [Math.floor(src / GRID), src % GRID];
      const [dr, dc] = [Math.floor(dst / GRID), dst % GRID];
      const d = Math.hypot(sr - dr, sc - dc);
      return Math.max(0, 0.42 - d * 0.12);
    }
    // A lesion patch attends strongly to the OTHER lesion, however far away.
    if (isLesion(dst)) return 0.85;
    const [sr, sc] = [Math.floor(src / GRID), src % GRID];
    const [dr, dc] = [Math.floor(dst / GRID), dst % GRID];
    const d = Math.hypot(sr - dr, sc - dc);
    return Math.max(0, 0.3 - d * 0.06);
  });
}

/**
 * Self-attention, shown as "what does this patch look at?".
 *
 * The comparison the prose is making — CNNs are local, Transformers are global —
 * is a claim about WHICH pixels influence a decision. So the interaction shows
 * exactly that: hover a patch and every other patch lights in proportion to the
 * attention it receives. Hover a lesion and the far-away second lesion lights up
 * despite the distance; hover healthy tissue and attention stays local.
 */
export function AttentionDiagram() {
  const [src, setSrc] = React.useState<number | null>(null);
  const weights = src === null ? null : attentionFrom(src);

  const explain =
    src === null
      ? null
      : isLesion(src)
        ? {
            title: "A lesion patch attends across the whole image",
            body:
              "This patch is looking hard at the other lesion in the opposite lung — despite the distance between them. Every patch is compared with every other patch in ONE step, so distance costs the model nothing. That is what a CNN cannot do in its early layers, where a filter only ever sees its own small neighbourhood.",
          }
        : {
            title: "A healthy patch has nothing to attend to",
            body:
              "Attention stays local and weak. Nothing here correlates with anything elsewhere, so the model spends no capacity on it. Attention is learnt, not fixed — the model decides what is worth relating to what.",
          };

  const cell = 30;
  const gap = 3;
  const size = GRID * cell + (GRID - 1) * gap;

  return (
    <DiagramFrame
      hint="Hover a patch to see what it attends to. Try a lesion (marked) and then some healthy tissue."
      explain={explain}
    >
      <div style={{ display: "flex", gap: "var(--sp-5)", alignItems: "center", flexWrap: "wrap" }}>
        <svg viewBox={`0 0 ${size} ${size + 18}`} style={{ width: size, minWidth: size, display: "block" }}>
          {Array.from({ length: GRID * GRID }, (_, i) => {
            const r = Math.floor(i / GRID);
            const c = i % GRID;
            const x = c * (cell + gap);
            const y = r * (cell + gap);
            const w = weights?.[i] ?? 0;
            const lesion = isLesion(i);
            return (
              <g key={i} onMouseEnter={() => setSrc(i)} onMouseLeave={() => setSrc(null)} style={{ cursor: "pointer" }}>
                <rect
                  x={x}
                  y={y}
                  width={cell}
                  height={cell}
                  rx={3}
                  fill={weights ? `color-mix(in srgb, var(--viz-1) ${Math.round(w * 100)}%, var(--surface))` : "var(--surface)"}
                  stroke={src === i ? "var(--primary)" : lesion ? "var(--viz-2)" : "var(--line-2)"}
                  strokeWidth={src === i ? 2.5 : lesion ? 1.6 : 1}
                  style={{ transition: "fill var(--dur-fast) var(--ease)" }}
                />
                {lesion && !weights && <circle cx={x + cell / 2} cy={y + cell / 2} r={3.5} fill="var(--viz-2)" opacity={0.75} />}
              </g>
            );
          })}
          <text x={0} y={size + 13} style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
            36 patches · outlined = lesion
          </text>
        </svg>

        {/* Capped: an uncapped flex column stretches the ramp legend to the full
            figure width, which reads as a chart rather than a key. */}
        <div style={{ flex: "1 1 190px", minWidth: "180px", maxWidth: "360px" }}>
          <div style={{ font: "600 11px var(--font-mono)", color: "var(--ink-3)", marginBottom: "0.5rem" }}>
            ATTENTION RECEIVED
          </div>
          {/* Sequential ramp = magnitude. One hue, light→dark, with a scale legend. */}
          <div
            style={{
              height: "10px",
              borderRadius: "3px",
              background: "linear-gradient(to right, var(--surface), var(--viz-1))",
              border: "1px solid var(--line-2)",
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.3rem" }}>
            <span style={{ font: "500 9px var(--font-mono)", color: "var(--ink-4)" }}>none</span>
            <span style={{ font: "500 9px var(--font-mono)", color: "var(--ink-4)" }}>strong</span>
          </div>
          <p
            style={{
              font: "400 12px var(--font-sans)",
              color: "var(--ink-3)",
              lineHeight: 1.45,
              margin: "0.85rem 0 0",
            }}
          >
            Every patch is compared with every other patch. That is{" "}
            <strong style={{ color: "var(--ink-2)" }}>36 × 36 = 1,296</strong> comparisons here — and it grows with the{" "}
            <em>square</em> of the patch count, which is why full-resolution ViTs get expensive, and why Swin and Mamba
            exist.
          </p>
        </div>
      </div>
    </DiagramFrame>
  );
}
