import React from "react";
import { Icon } from "@/components/Icon";
import { xray, type Xray } from "@/data/xrays";

const BASE = import.meta.env.BASE_URL;

// Lesion classes wear STATUS-style meaning here, not series identity: Active is
// the thing we are hunting, Obsolete is the thing we must not mistake for it.
const CLS_COLOR: Record<string, string> = {
  Active: "var(--viz-2)", // red
  Obsolete: "var(--viz-4)", // amber
};

/**
 * A chest X-ray with its radiologist ground truth as a toggleable SVG overlay.
 *
 * Boxes are drawn from data rather than burned into the image, which is what
 * makes the reveal possible: the reader gets to look first and find out how hard
 * this actually is before the answer appears. That is the whole point of showing
 * an X-ray on a page about automated detection.
 */
export function XrayView({
  slug,
  defaultShown = false,
  height = 300,
}: {
  slug: string;
  defaultShown?: boolean;
  height?: number;
}) {
  const x: Xray = xray(slug);
  const [shown, setShown] = React.useState(defaultShown);
  const hasBoxes = x.boxes.length > 0;

  return (
    <div
      style={{
        border: "1px solid var(--line)",
        borderRadius: "var(--r)",
        overflow: "hidden",
        background: "var(--surface)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ position: "relative", background: "#000", lineHeight: 0 }}>
        <svg
          viewBox={`0 0 ${x.w} ${x.h}`}
          style={{ width: "100%", height, display: "block", objectFit: "contain" }}
          role="img"
          aria-label={`${x.title}. ${
            hasBoxes ? `${x.boxes.length} annotated lesion${x.boxes.length > 1 ? "s" : ""}.` : "No lesions annotated."
          }`}
        >
          <image href={`${BASE}${x.file}`} x={0} y={0} width={x.w} height={x.h} preserveAspectRatio="xMidYMid meet" />
          {shown &&
            x.boxes.map((b, i) => {
              const [bx, by, bw, bh] = b.bbox;
              return (
                <g key={i}>
                  <rect
                    x={bx}
                    y={by}
                    width={bw}
                    height={bh}
                    fill="none"
                    stroke={CLS_COLOR[b.cls]}
                    strokeWidth={3}
                    rx={2}
                  />
                  <rect x={bx} y={Math.max(0, by - 19)} width={68} height={18} rx={2} fill={CLS_COLOR[b.cls]} />
                  <text
                    x={bx + 5}
                    y={Math.max(0, by - 19) + 13}
                    fill="#fff"
                    style={{ font: "600 11px var(--font-mono)" }}
                  >
                    {b.cls}
                  </text>
                </g>
              );
            })}
        </svg>

        {hasBoxes && (
          <button onClick={() => setShown((v) => !v)} className="xray-toggle" aria-pressed={shown}>
            <Icon name={shown ? "eye-off" : "eye"} size="0.85rem" />
            <span>{shown ? "Hide" : "Reveal"} lesions</span>
          </button>
        )}
        {!hasBoxes && (
          <span className="xray-badge">
            <Icon name="check" size="0.8rem" />
            <span>No TB lesions</span>
          </span>
        )}
      </div>

      <div style={{ padding: "0.75rem 0.9rem", borderTop: "1px solid var(--line)" }}>
        <div
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "var(--text-sm)",
            fontWeight: "var(--w-semibold)",
            color: "var(--ink)",
          }}
        >
          {x.title}
        </div>
        <div
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "var(--text-xs)",
            color: "var(--ink-3)",
            lineHeight: "var(--lh-snug)",
            marginTop: "0.25rem",
          }}
        >
          {x.caption}
        </div>
      </div>
    </div>
  );
}

/** Provenance line for any block of TBX11K figures. */
export function XraySource() {
  return (
    <>
      Images:{" "}
      <a
        href="https://github.com/yun-liu/Tuberculosis"
        target="_blank"
        rel="noreferrer"
        style={{ color: "var(--primary)", textDecoration: "none" }}
      >
        TBX11K
      </a>{" "}
      (Liu et al., CVPR 2020), CC BY 4.0. Boxes are the original radiologist annotations shipped with the dataset,
      not model predictions.
    </>
  );
}
