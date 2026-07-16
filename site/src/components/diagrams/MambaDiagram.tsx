import React from "react";
import { DiagramFrame } from "./primitives";

/**
 * Mamba vs Transformer, shown as the thing that actually distinguishes them:
 * how cost grows with sequence length.
 *
 * The claim in the prose — "linear complexity instead of quadratic", "~80% less
 * GPU memory" — is a claim about a CURVE. So the reader gets the curve, and gets
 * to drag the sequence length along it. A static "Mamba is more efficient" box
 * would be an assertion; this is the evidence.
 */
export function MambaDiagram() {
  const [n, setN] = React.useState(64);

  const W = 420;
  const H = 190;
  const PAD = { l: 42, r: 12, t: 12, b: 30 };
  const pw = W - PAD.l - PAD.r;
  const ph = H - PAD.t - PAD.b;

  const NMAX = 256;
  // Normalised cost curves. Quadratic is scaled to fill the plot at NMAX so the
  // shape is the message, not the absolute units.
  const quad = (x: number) => (x / NMAX) ** 2;
  const lin = (x: number) => x / NMAX;

  const px = (x: number) => PAD.l + (x / NMAX) * pw;
  const py = (v: number) => PAD.t + ph - v * ph;

  const path = (f: (x: number) => number) =>
    Array.from({ length: 65 }, (_, i) => {
      const x = (i / 64) * NMAX;
      return `${i === 0 ? "M" : "L"}${px(x).toFixed(1)} ${py(f(x)).toFixed(1)}`;
    }).join(" ");

  const ratio = n > 0 ? quad(n) / lin(n) : 0;

  return (
    <DiagramFrame
      hint="Drag the slider to grow the sequence and watch the two costs separate."
      explain={{
        title: `${n} patches — attention costs ${ratio < 1 ? "less than" : `${ratio.toFixed(1)}×`} ${ratio >= 1 ? "what the SSM costs" : ""}`,
        body:
          n <= 64
            ? "At short sequences the two are close, and the Transformer's global view is nearly free. This is the regime a 224×224 image with big patches sits in."
            : "As the sequence grows, self-attention's every-patch-against-every-patch cost grows with the square of the length while the SSM's grows linearly. This is exactly the regime a high-resolution chest X-ray lives in — and it is why Mamba can hold global context on hardware where a full ViT cannot.",
      }}
    >
      <div style={{ display: "flex", gap: "var(--sp-5)", flexWrap: "wrap", alignItems: "center" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ flex: "1 1 340px", minWidth: "320px", display: "block" }}>
          {/* Solid hairline gridlines — never dashed. */}
          {[0, 0.25, 0.5, 0.75, 1].map((g) => (
            <line key={g} x1={PAD.l} y1={py(g)} x2={W - PAD.r} y2={py(g)} stroke="var(--viz-grid)" strokeWidth={1} />
          ))}
          <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={PAD.t + ph} stroke="var(--viz-axis)" strokeWidth={1} />
          <line x1={PAD.l} y1={PAD.t + ph} x2={W - PAD.r} y2={PAD.t + ph} stroke="var(--viz-axis)" strokeWidth={1} />

          <path d={path(quad)} fill="none" stroke="var(--viz-2)" strokeWidth={2} />
          <path d={path(lin)} fill="none" stroke="var(--viz-1)" strokeWidth={2} />

          {/* current position */}
          <line x1={px(n)} y1={PAD.t} x2={px(n)} y2={PAD.t + ph} stroke="var(--ink-4)" strokeWidth={1} />
          <circle cx={px(n)} cy={py(quad(n))} r={4.5} fill="var(--viz-2)" stroke="var(--surface)" strokeWidth={2} />
          <circle cx={px(n)} cy={py(lin(n))} r={4.5} fill="var(--viz-1)" stroke="var(--surface)" strokeWidth={2} />

          {/* Direct labels at the endpoints — no legend box needed for two labelled series. */}
          <text x={W - PAD.r} y={py(quad(NMAX)) + 12} textAnchor="end" style={{ font: "600 10px var(--font-mono)", fill: "var(--viz-2)" }}>
            attention O(n²)
          </text>
          <text x={W - PAD.r} y={py(lin(NMAX)) - 6} textAnchor="end" style={{ font: "600 10px var(--font-mono)", fill: "var(--viz-1)" }}>
            SSM O(n)
          </text>

          <text x={PAD.l - 6} y={py(0) + 3} textAnchor="end" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
            0
          </text>
          <text x={(PAD.l + W - PAD.r) / 2} y={H - 8} textAnchor="middle" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
            sequence length (patches) →
          </text>
          <text
            x={12}
            y={PAD.t + ph / 2}
            textAnchor="middle"
            transform={`rotate(-90 12 ${PAD.t + ph / 2})`}
            style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}
          >
            compute / memory →
          </text>
        </svg>

        <div style={{ flex: "1 1 190px", minWidth: "180px" }}>
          <label style={{ font: "600 11px var(--font-mono)", color: "var(--ink-3)", display: "block", marginBottom: "0.5rem" }}>
            SEQUENCE LENGTH: <span style={{ color: "var(--primary)" }}>{n}</span>
          </label>
          <input
            type="range"
            min={8}
            max={NMAX}
            step={8}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--primary)" }}
            aria-label="Sequence length in patches"
          />
          <div
            style={{
              marginTop: "0.9rem",
              display: "grid",
              gap: "0.4rem",
              font: "500 11px var(--font-mono)",
              color: "var(--ink-2)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--ink-3)" }}>attention</span>
              <span style={{ fontVariantNumeric: "tabular-nums" }}>{(n * n).toLocaleString()} ops</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--ink-3)" }}>SSM</span>
              <span style={{ fontVariantNumeric: "tabular-nums" }}>{n.toLocaleString()} ops</span>
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                borderTop: "1px solid var(--line)",
                paddingTop: "0.4rem",
                fontWeight: 600,
              }}
            >
              <span style={{ color: "var(--ink-3)" }}>ratio</span>
              <span style={{ color: "var(--viz-2)", fontVariantNumeric: "tabular-nums" }}>{n}×</span>
            </div>
          </div>
        </div>
      </div>
    </DiagramFrame>
  );
}
