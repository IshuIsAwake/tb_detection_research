
export interface Bar {
  label: string;
  value: number;
  /** Overrides the formatted value at the bar end. */
  display?: string;
  /** Lift this bar out of the pack (emphasis form). */
  emphasis?: boolean;
  note?: string;
}

/**
 * Horizontal bar chart for magnitude comparison.
 *
 * Horizontal because every dataset here has long category names (WHO regions,
 * country names, architecture names) — rotated x-labels are the alternative and
 * they are worse.
 *
 * Colour: these are NOMINAL categories (countries, risk factors, models), so
 * every bar wears slot 1. Colouring each bar by its own value would re-encode
 * the bar length in hue and spend the identity channel on nothing. Where one bar
 * IS the story, `emphasis` keeps it in the accent and greys the rest.
 *
 * Values are direct-labelled at the bar end, so the tooltip is an enhancement
 * rather than the only way to read the chart.
 */
export function BarChart({
  bars,
  unit = "",
  max,
  labelWidth = 168,
}: {
  bars: Bar[];
  unit?: string;
  /** Defaults to the largest value; set to a scale ceiling (e.g. 100) if meaningful. */
  max?: number;
  labelWidth?: number;
}) {
  const ceiling = max ?? Math.max(...bars.map((b) => b.value));
  const anyEmphasis = bars.some((b) => b.emphasis);

  return (
    <div style={{ display: "grid", gap: "0.55rem" }}>
      {bars.map((b) => {
        const pct = ceiling > 0 ? (b.value / ceiling) * 100 : 0;
        // Emphasis form: the highlighted bar keeps the accent, the rest recede.
        const fill = !anyEmphasis || b.emphasis ? "var(--viz-1)" : "var(--ink-4)";
        return (
          <div
            key={b.label}
            className="bar-row"
            style={{ display: "grid", gridTemplateColumns: `${labelWidth}px 1fr`, gap: "0.75rem", alignItems: "center" }}
            title={`${b.label}: ${b.display ?? b.value + unit}${b.note ? " — " + b.note : ""}`}
          >
            <div
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-sm)",
                color: b.emphasis ? "var(--ink)" : "var(--ink-2)",
                fontWeight: b.emphasis ? "var(--w-semibold)" : "var(--w-regular)",
                textAlign: "right",
                lineHeight: 1.3,
                minWidth: 0,
              }}
            >
              {b.label}
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", minWidth: 0 }}>
              <div
                style={{
                  flex: 1,
                  height: "13px",
                  background: "var(--viz-track)",
                  borderRadius: "3px",
                  overflow: "hidden",
                  minWidth: 0,
                }}
              >
                <div
                  style={{
                    width: `${pct}%`,
                    height: "100%",
                    background: fill,
                    // Rounded data-end only; the baseline end stays square so the
                    // bar reads as anchored to zero.
                    borderRadius: "0 3px 3px 0",
                    transition: "width var(--dur-slow) var(--ease)",
                  }}
                />
              </div>
              <div
                style={{
                  flexShrink: 0,
                  fontFamily: "var(--font-mono)",
                  fontSize: "var(--text-xs)",
                  fontWeight: "var(--w-semibold)",
                  color: b.emphasis || !anyEmphasis ? "var(--ink)" : "var(--ink-3)",
                  fontVariantNumeric: "tabular-nums",
                  minWidth: "3.2rem",
                }}
              >
                {b.display ?? `${b.value}${unit}`}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
