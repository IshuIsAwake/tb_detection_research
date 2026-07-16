import React from "react";

export interface Segment {
  label: string;
  value: number;
  /** 1-based categorical slot. Assigned in order, never cycled past 6. */
  slot: 1 | 2 | 3 | 4 | 5 | 6;
}

const slotVar = (s: number) => `var(--viz-${s})`;

/**
 * Part-to-whole as a single horizontal stacked bar.
 *
 * A donut was the obvious reach here and it is the wrong form: the top three WHO
 * regions are 34 / 27 / 25 percent, and arcs that close are exactly what people
 * cannot compare. A horizontal stack keeps the part-to-whole reading while
 * leaving the segments comparable, and it copes with long region names.
 *
 * Segments are separated by a 2px surface gap rather than a border — a stroke
 * around each fill would add a third colour to every boundary.
 */
export function StackedBar({
  segments,
  unit = "%",
  height = 30,
}: {
  segments: Segment[];
  unit?: string;
  height?: number;
}) {
  const [hover, setHover] = React.useState<string | null>(null);
  const total = segments.reduce((a, s) => a + s.value, 0);

  return (
    <div>
      <div
        style={{
          display: "flex",
          gap: "2px",
          height,
          borderRadius: "4px",
          overflow: "hidden",
          background: "var(--viz-track)",
        }}
        role="img"
        aria-label={segments.map((s) => `${s.label} ${s.value}${unit}`).join(", ")}
      >
        {segments.map((s) => {
          const pct = (s.value / total) * 100;
          const dim = hover !== null && hover !== s.label;
          return (
            <div
              key={s.label}
              onMouseEnter={() => setHover(s.label)}
              onMouseLeave={() => setHover(null)}
              title={`${s.label}: ${s.value}${unit}`}
              style={{
                width: `${pct}%`,
                background: slotVar(s.slot),
                opacity: dim ? 0.35 : 1,
                transition: "opacity var(--dur-fast) var(--ease)",
                cursor: "default",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                minWidth: 0,
              }}
            >
              {/* Only label inside the segment when it comfortably fits. */}
              {pct >= 12 && (
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.68rem",
                    fontWeight: "var(--w-semibold)",
                    color: "#fff",
                    fontVariantNumeric: "tabular-nums",
                    whiteSpace: "nowrap",
                  }}
                >
                  {s.value}
                  {unit}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend — always present for >= 2 series, so identity is never colour-alone. */}
      <ul
        style={{
          listStyle: "none",
          margin: "0.85rem 0 0",
          padding: 0,
          display: "flex",
          flexWrap: "wrap",
          gap: "0.4rem 1.1rem",
        }}
      >
        {segments.map((s) => {
          const dim = hover !== null && hover !== s.label;
          return (
            <li
              key={s.label}
              onMouseEnter={() => setHover(s.label)}
              onMouseLeave={() => setHover(null)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
                opacity: dim ? 0.45 : 1,
                transition: "opacity var(--dur-fast) var(--ease)",
                cursor: "default",
              }}
            >
              <span
                style={{
                  width: "9px",
                  height: "9px",
                  borderRadius: "2px",
                  background: slotVar(s.slot),
                  flexShrink: 0,
                }}
              />
              {/* Text wears ink tokens, never the series colour — the swatch carries identity. */}
              <span style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-xs)", color: "var(--ink-2)" }}>
                {s.label}
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "var(--text-xs)",
                  color: "var(--ink-3)",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {s.value}
                {unit}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
