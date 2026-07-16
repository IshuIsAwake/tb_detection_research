import React from "react";
import { DiagramFrame, Block, Arrow, ArrowDefs, Caption } from "./primitives";

type Part = "input" | "early" | "mid" | "deep" | "head" | null;

const EXPLAIN: Record<Exclude<Part, null>, { title: string; body: string }> = {
  input: {
    title: "Raw pixels — 512 × 512",
    body:
      "The network is handed the chest X-ray as a grid of intensity values and nothing else. No radiologist tells it what a cavity looks like; there is no list of features to measure. This is the whole break from the 2000s pipeline, which needed a human to define 'circularity' or 'Gabor texture frequency' by hand.",
  },
  early: {
    title: "Early layers — edges & gradients",
    body:
      "The first convolutions slide small kernels across the image and fire on the simplest things present: edges, gradients, lines. These filters are near-universal — the ones learnt on photographs of cats transfer to chest X-rays almost unchanged, which is exactly why ImageNet pretraining works on medical images at all.",
  },
  mid: {
    title: "Middle layers — textures & shapes",
    body:
      "Edges get composed into textures and shapes: the granular mottle of miliary TB, the ring of a cavity wall, the sharp margin of a calcified granuloma. Each layer sees a wider patch of the original image than the one before it, so the features get progressively less local.",
  },
  deep: {
    title: "Deep layers — pathology-level patterns",
    body:
      "The deepest layers assemble whole disease-specific patterns. Crucially some of these do not map onto any named radiological sign — the network is free to find whatever correlates with disease, rather than being confined to the vocabulary a human could write down.",
  },
  head: {
    title: "Head — the decision",
    body:
      "The final layers collapse the feature map into an answer: a probability for the classifier, or box coordinates plus a class for a detector. Everything upstream is shared; swapping this head is what turns a classifier into a detector.",
  },
};

/**
 * The hierarchical-feature idea, which is the single thing a reader has to
 * understand before any of the later architectures make sense: a CNN is not
 * given features, it builds them, and it builds them in layers of increasing
 * abstraction.
 *
 * The blocks shrink and deepen left-to-right because that is literally what
 * happens to the tensor — spatial dimensions fall, channel count rises.
 */
export function CNNDiagram() {
  const [part, setPart] = React.useState<Part>(null);
  const on = (p: Part) => ({ onMouseEnter: () => setPart(p), onMouseLeave: () => setPart(null) });
  const dim = (p: Part) => part !== null && part !== p;

  return (
    <DiagramFrame
      hint="Hover any stage to see what the network has learnt by that point."
      explain={part ? EXPLAIN[part] : null}
    >
      <svg viewBox="0 0 660 172" style={{ width: "100%", minWidth: "560px", display: "block" }}>
        <ArrowDefs />

        {/* Input image */}
        <g {...on("input")} style={{ cursor: "pointer", opacity: dim("input") ? 0.32 : 1 }}>
          <rect x={8} y={38} width={68} height={68} rx={4} fill="#2a2e33" stroke={part === "input" ? "var(--primary)" : "var(--line-2)"} strokeWidth={part === "input" ? 2 : 1} />
          {/* a suggestion of a ribcage, not a real X-ray */}
          <path d="M26 56 q16 -9 32 0 M24 68 q18 -10 36 0 M24 80 q18 -10 36 0 M42 50 v46" stroke="#6d757d" strokeWidth={1.5} fill="none" strokeLinecap="round" />
          <Caption x={42} y={122}>512×512×1</Caption>
        </g>
        <Arrow x1={80} y1={72} x2={100} y2={72} active={part === "early"} dim={part !== null && part !== "early"} />

        {/* Conv stages — spatial shrinks, channels grow */}
        <Block x={104} y={44} w={62} h={56} label="conv" sub="256²×32" active={part === "early"} dim={dim("early")} {...on("early")} />
        <Caption x={135} y={122}>edges</Caption>
        <Arrow x1={170} y1={72} x2={190} y2={72} active={part === "mid"} dim={part !== null && part !== "mid"} />

        <Block x={194} y={52} w={62} h={40} label="conv" sub="128²×64" active={part === "mid"} dim={dim("mid")} {...on("mid")} />
        <Caption x={225} y={122}>textures</Caption>
        <Arrow x1={260} y1={72} x2={280} y2={72} active={part === "mid"} dim={part !== null && part !== "mid"} />

        <Block x={284} y={56} w={62} h={32} label="conv" sub="64²×128" active={part === "mid"} dim={dim("mid")} {...on("mid")} />
        <Caption x={315} y={122}>shapes</Caption>
        <Arrow x1={350} y1={72} x2={370} y2={72} active={part === "deep"} dim={part !== null && part !== "deep"} />

        <Block x={374} y={60} w={62} h={24} label="conv" sub="32²×256" active={part === "deep"} dim={dim("deep")} {...on("deep")} />
        <Caption x={405} y={122}>lesions</Caption>
        <Arrow x1={440} y1={72} x2={460} y2={72} active={part === "head"} dim={part !== null && part !== "head"} />

        {/* Head */}
        <Block
          x={464}
          y={46}
          w={92}
          h={52}
          label="head"
          sub="class / box"
          fill="var(--primary-tint)"
          stroke="var(--primary-tint-2)"
          active={part === "head"}
          dim={dim("head")}
          {...on("head")}
        />
        <Arrow x1={560} y1={72} x2={580} y2={72} active={part === "head"} dim={part !== null && part !== "head"} />
        <text x={588} y={68} style={{ font: "600 11px var(--font-mono)", fill: "var(--viz-2)" }}>
          TB
        </text>
        <text x={588} y={82} style={{ font: "500 10px var(--font-mono)", fill: "var(--ink-3)" }}>
          0.94
        </text>

        {/* Axis of abstraction */}
        <line x1={8} y1={148} x2={556} y2={148} stroke="var(--line-2)" strokeWidth={1} />
        <text x={8} y={164} style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
          local, generic
        </text>
        <text x={556} y={164} textAnchor="end" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
          global, disease-specific
        </text>
      </svg>
    </DiagramFrame>
  );
}
