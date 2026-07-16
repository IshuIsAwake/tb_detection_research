import React from "react";
import { DiagramFrame, Block, Arrow, ArrowDefs, useReducedMotion } from "./primitives";

type Part = "main" | "skip" | "add" | null;

const EXPLAIN: Record<Exclude<Part, null>, { title: string; body: string }> = {
  main: {
    title: "The residual branch — F(x)",
    body:
      "Two stacked convolutions. On their own, this is just a plain deep network — and plain deep networks get WORSE past a certain depth, because the gradient has to survive multiplication through every layer on its way back and shrinks toward zero.",
  },
  skip: {
    title: "The skip connection — the whole idea",
    body:
      "The input is carried around the convolutions untouched and added back on. On the backward pass this is a clean road: the gradient reaches earlier layers without being multiplied through the convolutions, so it never vanishes. This one wire is what made 50-, 101-, and 152-layer networks trainable.",
  },
  add: {
    title: "F(x) + x — learning the difference",
    body:
      "Because the output is F(x) + x, the layers only have to learn the RESIDUAL — how much to CHANGE x, not how to rebuild it. If a block has nothing useful to add it can drive F(x) toward zero and pass x straight through. That is why adding depth can no longer make training error worse: extra blocks can always fall back to doing nothing.",
  },
};

/**
 * The skip connection, which is the single most-cited idea this project depends
 * on — YOLOv8's backbone is residual, and every ResNet row in the comparison
 * table is downstream of this wire.
 *
 * The interaction is doing real work here: hovering the skip animates the
 * gradient travelling BACKWARD along it, because "the gradient flows straight
 * through" is a claim about direction that a static arrow cannot make.
 */
export function ResNetDiagram() {
  const [part, setPart] = React.useState<Part>(null);
  const reduce = useReducedMotion();
  const on = (p: Part) => ({ onMouseEnter: () => setPart(p), onMouseLeave: () => setPart(null) });
  const dim = (p: Part) => part !== null && part !== p;

  const skipActive = part === "skip" || part === "add";

  return (
    <DiagramFrame
      hint="Hover the skip connection to watch the gradient travel back through it."
      explain={part ? EXPLAIN[part] : null}
    >
      <svg viewBox="0 0 620 200" style={{ width: "100%", minWidth: "520px", display: "block" }}>
        <ArrowDefs />

        {/* input */}
        <circle cx={34} cy={96} r={4} fill="var(--ink-3)" />
        <text x={34} y={80} textAnchor="middle" style={{ font: "italic 600 13px var(--font-serif)", fill: "var(--ink)" }}>
          x
        </text>
        <Arrow x1={40} y1={96} x2={72} y2={96} />

        {/* main branch */}
        <g {...on("main")} style={{ cursor: "pointer" }}>
          <rect
            x={76}
            y={62}
            width={286}
            height={68}
            rx={6}
            fill="none"
            stroke={part === "main" ? "var(--primary)" : "transparent"}
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={dim("main") ? 0.3 : 1}
          />
        </g>
        <Block x={86} y={74} w={104} h={44} label="conv 3×3" sub="BN → ReLU" active={part === "main"} dim={dim("main")} {...on("main")} />
        <Arrow x1={194} y1={96} x2={222} y2={96} dim={dim("main")} />
        <Block x={226} y={74} w={104} h={44} label="conv 3×3" sub="BN" active={part === "main"} dim={dim("main")} {...on("main")} />
        <Arrow x1={334} y1={96} x2={392} y2={96} dim={dim("main")} />
        <text x={219} y={148} textAnchor="middle" style={{ font: "italic 600 12px var(--font-serif)", fill: part === "main" ? "var(--primary)" : "var(--ink-3)" }}>
          F(x)
        </text>

        {/* skip branch */}
        <g {...on("skip")} style={{ cursor: "pointer" }}>
          {/* Fat invisible hit area — a 1.5px wire is an unhittable target.
              pointerEvents="stroke" is required: SVG's default visiblePainted
              hit-testing can skip a zero-alpha stroke entirely. */}
          <path
            d="M56 96 L56 32 L404 32 L404 84"
            stroke="transparent"
            strokeWidth={20}
            fill="none"
            pointerEvents="stroke"
          />
          <path
            d="M56 96 L56 32 L404 32 L404 84"
            stroke={skipActive ? "var(--primary)" : "var(--line-3)"}
            strokeWidth={skipActive ? 2.5 : 1.5}
            fill="none"
            markerEnd={`url(#${skipActive ? "arrow-active" : "arrow"})`}
            opacity={part === "main" ? 0.3 : 1}
            style={{ transition: "stroke var(--dur-fast) var(--ease), stroke-width var(--dur-fast) var(--ease)" }}
          />
          <text
            x={230}
            y={24}
            textAnchor="middle"
            style={{
              font: "600 10px var(--font-mono)",
              fill: skipActive ? "var(--primary)" : "var(--ink-3)",
              opacity: part === "main" ? 0.3 : 1,
            }}
          >
            identity — x carried around untouched
          </text>

          {/* the gradient, travelling BACKWARD along the skip */}
          {skipActive && !reduce && (
            <circle r={4.5} fill="var(--primary)">
              <animateMotion dur="1.6s" repeatCount="indefinite" path="M404 84 L404 32 L56 32 L56 96" />
            </circle>
          )}
          {skipActive && (
            <text
              x={230}
              y={46}
              textAnchor="middle"
              style={{ font: "600 9px var(--font-mono)", fill: "var(--primary)" }}
            >
              ← gradient, unmultiplied
            </text>
          )}
        </g>

        {/* add */}
        <g {...on("add")} style={{ cursor: "pointer" }}>
          <circle
            cx={404}
            cy={96}
            r={13}
            fill="var(--surface)"
            stroke={part === "add" ? "var(--primary)" : "var(--line-2)"}
            strokeWidth={part === "add" ? 2 : 1}
          />
          <path d="M398 96 h12 M404 90 v12" stroke="var(--ink)" strokeWidth={1.6} strokeLinecap="round" />
        </g>
        <Arrow x1={419} y1={96} x2={446} y2={96} active={part === "add"} />

        <Block x={450} y={74} w={72} h={44} label="ReLU" active={part === "add"} dim={dim("add")} {...on("add")} />
        <Arrow x1={526} y1={96} x2={556} y2={96} />
        <text x={578} y={101} textAnchor="middle" style={{ font: "italic 600 13px var(--font-serif)", fill: "var(--ink)" }}>
          H(x)
        </text>

        {/* the equation, which is the punchline */}
        <text
          x={310}
          y={182}
          textAnchor="middle"
          style={{
            font: "italic 600 15px var(--font-serif)",
            fill: part === "add" ? "var(--primary)" : "var(--ink-2)",
            transition: "fill var(--dur-fast) var(--ease)",
          }}
        >
          H(x) = F(x) + x
        </text>
        <text x={310} y={196} textAnchor="middle" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
          the block learns the residual, not the whole mapping
        </text>
      </svg>
    </DiagramFrame>
  );
}
