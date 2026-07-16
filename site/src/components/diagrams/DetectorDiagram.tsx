import React from "react";
import { DiagramFrame, Block, Arrow, ArrowDefs } from "./primitives";

type Which = "two" | "one";

/**
 * One-stage vs two-stage detection — the choice this project actually made.
 *
 * A tab rather than two static figures, because the two pipelines are the same
 * diagram with a different middle: showing them in the same frame makes the
 * structural difference (a proposal step, or not) the thing you notice.
 */
export function DetectorDiagram() {
  const [which, setWhich] = React.useState<Which>("two");

  return (
    <div>
      <div role="tablist" aria-label="Detector family" style={{ display: "flex", gap: "0.4rem", marginBottom: "0.7rem" }}>
        {(
          [
            ["two", "Two-stage — Faster R-CNN"],
            ["one", "One-stage — YOLO"],
          ] as [Which, string][]
        ).map(([k, label]) => (
          <button
            key={k}
            role="tab"
            aria-selected={which === k}
            onClick={() => setWhich(k)}
            style={{
              all: "unset",
              cursor: "pointer",
              padding: "0.35rem 0.75rem",
              borderRadius: "var(--r-pill)",
              fontFamily: "var(--font-sans)",
              fontSize: "var(--text-xs)",
              fontWeight: "var(--w-semibold)",
              color: which === k ? "var(--primary)" : "var(--ink-3)",
              background: which === k ? "var(--primary-tint)" : "transparent",
              border: `1px solid ${which === k ? "var(--primary-tint-2)" : "var(--line)"}`,
              transition: "background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease)",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <DiagramFrame
        hint=""
        explain={
          which === "two"
            ? {
                title: "Two-stage: propose, then classify",
                body:
                  "A Region Proposal Network first asks only 'is there SOMETHING here?' and emits candidate boxes. A second head then asks 'what is it, and exactly where?'. Two passes over the region means high precision and classically strong small-lesion recall — at a real speed cost. Xie et al. reached AUC 0.977 on Shenzhen + Montgomery this way.",
              }
            : {
                title: "One-stage: predict boxes and classes in one pass",
                body:
                  "The image is divided into a grid, and every cell regresses box coordinates and class probabilities simultaneously — detection as a single regression problem. No proposal step, so it is fast enough for real time. This is the family this project uses: YOLOv8n, and the reason the whole detector trains in 15 minutes on one consumer GPU.",
              }
        }
      >
        <svg viewBox="0 0 620 132" style={{ width: "100%", minWidth: "540px", display: "block" }}>
          <ArrowDefs />

          {/* input */}
          <rect x={8} y={40} width={50} height={50} rx={4} fill="#2a2e33" stroke="var(--line-2)" />
          <path d="M20 56 q13 -7 26 0 M19 68 q14 -8 28 0 M33 50 v34" stroke="#6d757d" strokeWidth={1.2} fill="none" strokeLinecap="round" />
          <text x={33} y={104} textAnchor="middle" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
            CXR
          </text>
          <Arrow x1={62} y1={65} x2={86} y2={65} />

          <Block x={90} y={43} w={82} h={44} label="backbone" sub="CNN" />

          {which === "two" ? (
            <>
              <Arrow x1={176} y1={65} x2={202} y2={65} active />
              <Block
                x={206}
                y={43}
                w={104}
                h={44}
                label="RPN"
                sub="where might it be?"
                fill="var(--primary-tint)"
                stroke="var(--primary-tint-2)"
                active
              />
              <Arrow x1={314} y1={65} x2={340} y2={65} active />
              <Block x={344} y={43} w={104} h={44} label="RoI head" sub="what is it?" fill="var(--primary-tint)" stroke="var(--primary-tint-2)" active />
              <Arrow x1={452} y1={65} x2={478} y2={65} />
              <text x={382} y={112} textAnchor="middle" style={{ font: "600 9px var(--font-mono)", fill: "var(--viz-2)" }}>
                two passes — precise, slow
              </text>
            </>
          ) : (
            <>
              <Arrow x1={176} y1={65} x2={264} y2={65} active />
              <Block
                x={268}
                y={43}
                w={180}
                h={44}
                label="dense prediction head"
                sub="box + class, every cell, one pass"
                fill="var(--primary-tint)"
                stroke="var(--primary-tint-2)"
                active
              />
              <Arrow x1={452} y1={65} x2={478} y2={65} />
              <text x={358} y={112} textAnchor="middle" style={{ font: "600 9px var(--font-mono)", fill: "var(--viz-1)" }}>
                one pass — fast, this project's choice
              </text>
            </>
          )}

          {/* output: image with a box on it */}
          <rect x={482} y={40} width={50} height={50} rx={4} fill="#2a2e33" stroke="var(--line-2)" />
          <path d="M494 56 q13 -7 26 0 M493 68 q14 -8 28 0 M507 50 v34" stroke="#6d757d" strokeWidth={1.2} fill="none" strokeLinecap="round" />
          <rect x={492} y={50} width={20} height={20} fill="none" stroke="var(--viz-2)" strokeWidth={1.8} rx={1} />
          <text x={545} y={62} style={{ font: "600 10px var(--font-mono)", fill: "var(--viz-2)" }}>
            Active
          </text>
          <text x={545} y={75} style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-3)" }}>
            0.94
          </text>
        </svg>
      </DiagramFrame>
    </div>
  );
}
