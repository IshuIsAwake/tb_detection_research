
export interface Era {
  id: string;
  years: string;
  title: string;
  /** Short enough to read in the rail; the prose carries the detail. */
  blurb: string;
  /** Marks a turning point — the field changing its mind. */
  pivot?: boolean;
}

/**
 * A vertical timeline for the history section.
 *
 * Deliberately not a chart: the data is a sequence of named eras, not a
 * magnitude, so there is nothing to encode in length or colour. The accent is
 * reserved for the pivots — the moments the field reversed itself — because
 * those are the argument the section is making.
 */
export function Timeline({ eras }: { eras: Era[] }) {
  return (
    <ol style={{ listStyle: "none", margin: 0, padding: 0, position: "relative" }}>
      {/* The spine */}
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          left: "5.9rem",
          top: "0.6rem",
          bottom: "0.6rem",
          width: "1px",
          background: "var(--line-2)",
        }}
      />
      {eras.map((e) => (
        <li
          key={e.id}
          style={{
            display: "grid",
            gridTemplateColumns: "5.2rem 1.4rem 1fr",
            gap: "0 0.5rem",
            alignItems: "start",
            paddingBottom: "var(--sp-5)",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.72rem",
              fontWeight: "var(--w-semibold)",
              color: e.pivot ? "var(--primary)" : "var(--ink-3)",
              textAlign: "right",
              paddingTop: "0.15rem",
              whiteSpace: "nowrap",
            }}
          >
            {e.years}
          </div>

          <div style={{ display: "flex", justifyContent: "center", paddingTop: "0.3rem" }}>
            <span
              style={{
                width: e.pivot ? "11px" : "7px",
                height: e.pivot ? "11px" : "7px",
                borderRadius: "50%",
                background: e.pivot ? "var(--primary)" : "var(--line-3)",
                // A surface ring keeps the dot readable where it overlaps the spine.
                boxShadow: `0 0 0 3px var(--paper)`,
                zIndex: 1,
              }}
            />
          </div>

          <div style={{ minWidth: 0, paddingTop: "0.05rem" }}>
            <div
              style={{
                fontFamily: "var(--font-serif)",
                fontSize: "1rem",
                fontWeight: "var(--w-semibold)",
                color: "var(--ink)",
                lineHeight: "var(--lh-snug)",
              }}
            >
              {e.title}
            </div>
            <div
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-sm)",
                color: "var(--ink-2)",
                lineHeight: "var(--lh-snug)",
                marginTop: "0.25rem",
                maxWidth: "58ch",
              }}
            >
              {e.blurb}
            </div>
          </div>
        </li>
      ))}
    </ol>
  );
}
