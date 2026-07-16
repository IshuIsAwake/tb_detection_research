import React from "react";
import { DiagramFrame, Block, Arrow, ArrowDefs } from "./primitives";

type Part = "down" | "bottom" | "up" | "skip" | null;

const EXPLAIN: Record<Exclude<Part, null>, { title: string; body: string }> = {
  down: {
    title: "Contracting path — what is here?",
    body:
      "Standard convolutions and pooling. Spatial resolution falls at every step while the channel count rises: the network learns what it is looking at, and progressively forgets exactly where. Same shape as the CNN classifier above — because it is one.",
  },
  bottom: {
    title: "The bottleneck — maximum context, minimum detail",
    body:
      "The narrowest point. The network now has rich semantic understanding of the whole image but has thrown away the fine spatial precision needed to draw a boundary. A classifier would stop here and emit a label. A segmenter cannot: it owes an answer for every single pixel.",
  },
  up: {
    title: "Expanding path — putting the where back",
    body:
      "Transposed convolutions upsample back toward full resolution, producing a pixel-by-pixel mask rather than one label for the image.",
  },
  skip: {
    title: "Skip connections — the reason it works",
    body:
      "Each upsampling step is handed the matching high-resolution feature map from the way down, concatenated straight on. The fine detail that pooling destroyed is reintroduced from the encoder, so the decoder gets deep semantics AND sharp edges. Same trick as ResNet — carry information around the lossy part — used for a different purpose.",
  },
};

/**
 * U-Net, drawn as the U it is named after.
 *
 * The skip connections are the load-bearing idea and they are hoverable as a
 * group, because they act as a group: it's the whole ladder that lets the
 * decoder recover spatial precision, not any single rung.
 */
export function UNetDiagram() {
  const [part, setPart] = React.useState<Part>(null);
  const on = (p: Part) => ({ onMouseEnter: () => setPart(p), onMouseLeave: () => setPart(null) });
  const dim = (p: Part) => part !== null && part !== p;

  // (x, y, height) for the four encoder levels; decoder mirrors on the right.
  const L = [
    { x: 60, y: 24, h: 76, sub: "512²" },
    { x: 122, y: 40, h: 52, sub: "256²" },
    { x: 184, y: 54, h: 34, sub: "128²" },
  ];
  const R = [
    { x: 372, y: 54, h: 34, sub: "128²" },
    { x: 434, y: 40, h: 52, sub: "256²" },
    { x: 496, y: 24, h: 76, sub: "512²" },
  ];

  return (
    <DiagramFrame hint="Hover the skip connections — they are the reason a U-Net can draw a sharp boundary." explain={part ? EXPLAIN[part] : null}>
      <svg viewBox="0 0 620 168" style={{ width: "100%", minWidth: "540px", display: "block" }}>
        <ArrowDefs />

        {/* encoder */}
        {L.map((l, i) => (
          <g key={i}>
            <Block x={l.x} y={l.y} w={44} h={l.h} label="conv" sub={l.sub} active={part === "down"} dim={dim("down")} {...on("down")} />
            {i < L.length - 1 && <Arrow x1={l.x + 44} y1={l.y + l.h / 2} x2={L[i + 1].x - 4} y2={L[i + 1].y + L[i + 1].h / 2} dim={dim("down")} />}
          </g>
        ))}
        <Arrow x1={228} y1={71} x2={264} y2={71} dim={dim("bottom")} />
        <text x={22} y={68} textAnchor="middle" style={{ font: "600 9px var(--font-mono)", fill: "var(--ink-3)" }}>
          image
        </text>
        <Arrow x1={38} y1={62} x2={56} y2={62} />

        {/* bottleneck */}
        <Block
          x={268}
          y={58}
          w={60}
          h={26}
          label="bottleneck"
          sub="64²"
          fill="var(--primary-tint)"
          stroke="var(--primary-tint-2)"
          active={part === "bottom"}
          dim={dim("bottom")}
          {...on("bottom")}
        />
        <Arrow x1={332} y1={71} x2={368} y2={71} dim={dim("up")} />

        {/* decoder */}
        {R.map((r, i) => (
          <g key={i}>
            <Block x={r.x} y={r.y} w={44} h={r.h} label="up" sub={r.sub} active={part === "up"} dim={dim("up")} {...on("up")} />
            {i < R.length - 1 && <Arrow x1={r.x + 44} y1={r.y + r.h / 2} x2={R[i + 1].x - 4} y2={R[i + 1].y + R[i + 1].h / 2} dim={dim("up")} />}
          </g>
        ))}
        <Arrow x1={544} y1={62} x2={562} y2={62} />
        <text x={584} y={66} textAnchor="middle" style={{ font: "600 9px var(--font-mono)", fill: "var(--ink-3)" }}>
          mask
        </text>

        {/* skips — the ladder across the top */}
        <g {...on("skip")} style={{ cursor: "pointer" }}>
          {L.map((l, i) => {
            const r = R[R.length - 1 - i];
            const y = l.y - 6 - i * 0;
            const active = part === "skip";
            return (
              <g key={i}>
                <path
                  d={`M${l.x + 22} ${l.y} L${l.x + 22} ${y - 4 - i * 5} L${r.x + 22} ${y - 4 - i * 5} L${r.x + 22} ${r.y}`}
                  stroke="transparent"
                  strokeWidth={14}
                  fill="none"
                  pointerEvents="stroke"
                />
                <path
                  d={`M${l.x + 22} ${l.y} L${l.x + 22} ${y - 4 - i * 5} L${r.x + 22} ${y - 4 - i * 5} L${r.x + 22} ${r.y}`}
                  stroke={active ? "var(--primary)" : "var(--line-3)"}
                  strokeWidth={active ? 2 : 1.2}
                  fill="none"
                  strokeDasharray={active ? "none" : "4 3"}
                  markerEnd={`url(#${active ? "arrow-active" : "arrow"})`}
                  opacity={part !== null && part !== "skip" ? 0.25 : 1}
                  style={{ transition: "stroke var(--dur-fast) var(--ease)" }}
                />
              </g>
            );
          })}
          <text
            x={298}
            y={12}
            textAnchor="middle"
            style={{
              font: "600 10px var(--font-mono)",
              fill: part === "skip" ? "var(--primary)" : "var(--ink-3)",
              opacity: part !== null && part !== "skip" ? 0.25 : 1,
            }}
          >
            skip connections — fine detail copied across
          </text>
        </g>

        <text x={144} y={158} textAnchor="middle" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
          resolution ↓ · semantics ↑
        </text>
        <text x={456} y={158} textAnchor="middle" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
          resolution ↑ · detail restored
        </text>
      </svg>
    </DiagramFrame>
  );
}
