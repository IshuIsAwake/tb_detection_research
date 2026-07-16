import { Icon } from "@/components/Icon";

export interface Stage {
  label: string;
  /** Absolute headcount. */
  value: number;
  display: string;
  note?: string;
  /** Label for the step FROM the previous stage TO this one. */
  step?: string;
}

/**
 * An ordered cascade: latent infection → active disease → death.
 *
 * Stages are ORDINAL (swapping them would change the meaning), so they take the
 * one-hue teal ramp rather than categorical slots — the reader should see the
 * order in the colour.
 *
 * The honesty problem: these stages span four orders of magnitude (2e9 → 1.23e6).
 * On a linear scale the last stage is 0.06% of the first and renders as an
 * invisible sliver, which reads as "nothing happens here" — the opposite of the
 * truth. So the bars are LOG-scaled and every step is direct-labelled with its
 * real conversion rate, and the caption says so. The numbers carry the magnitude;
 * the bars only carry the ordering.
 */
export function Cascade({ stages }: { stages: Stage[] }) {
  const logs = stages.map((s) => Math.log10(s.value));
  const lo = Math.min(...logs);
  const hi = Math.max(...logs);
  // Floor the shortest bar at 22% so the smallest stage stays a readable mark.
  const width = (v: number) => 22 + ((Math.log10(v) - lo) / (hi - lo)) * 78;
  const ramp = (i: number) => `var(--viz-ramp-${Math.min(5, 1 + i * 2)})`;

  return (
    <div>
      {stages.map((s, i) => (
        <div key={s.label}>
          {s.step && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.45rem",
                padding: "0.4rem 0 0.4rem 0.6rem",
                color: "var(--ink-3)",
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-xs)",
              }}
            >
              <Icon name="arrow-right" size="0.8rem" style={{ transform: "rotate(90deg)", opacity: 0.6 }} />
              <span>{s.step}</span>
            </div>
          )}
          <div
            style={{
              width: `${width(s.value)}%`,
              minWidth: "min(100%, 12rem)",
              background: ramp(i),
              borderRadius: "4px",
              padding: "0.7rem 0.9rem",
              display: "flex",
              alignItems: "baseline",
              gap: "0.7rem",
              flexWrap: "wrap",
              transition: "width var(--dur-slow) var(--ease)",
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "1.25rem",
                fontWeight: "var(--w-bold)",
                color: "#fff",
                letterSpacing: "var(--ls-tight)",
              }}
            >
              {s.display}
            </span>
            <span
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-sm)",
                color: "rgba(255,255,255,0.92)",
                fontWeight: "var(--w-medium)",
              }}
            >
              {s.label}
            </span>
          </div>
          {s.note && (
            <div
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-xs)",
                color: "var(--ink-3)",
                padding: "0.35rem 0 0 0.1rem",
                lineHeight: "var(--lh-snug)",
                maxWidth: "46ch",
              }}
            >
              {s.note}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
