import React from "react";

// Shared scaffolding for the architecture diagrams.
//
// House rules for every diagram in this folder:
//   - Inline SVG, no dependencies, no layout libraries.
//   - Interaction must TEACH, never merely decorate: hovering a part explains
//     the mechanism, it doesn't just tint it.
//   - Everything must be readable with interaction never used at all — the
//     default state is the complete picture, hover only adds emphasis.
//   - Honour prefers-reduced-motion.

export function useReducedMotion() {
  const [reduce, setReduce] = React.useState(false);
  React.useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduce(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduce;
}

/** Frame + explain-panel that every diagram sits in. */
export function DiagramFrame({
  children,
  hint,
  explain,
}: {
  children: React.ReactNode;
  /** Shown when nothing is hovered — tells the reader what to do. */
  hint: string;
  /** Shown when a part is hovered/selected. */
  explain?: { title: string; body: string } | null;
}) {
  return (
    <div>
      <div
        style={{
          background: "var(--paper-2)",
          border: "1px solid var(--line)",
          borderRadius: "var(--r)",
          padding: "var(--sp-4)",
          overflowX: "auto",
        }}
      >
        {children}
      </div>
      <div
        style={{
          marginTop: "0.6rem",
          minHeight: "3.4rem",
          padding: "0.7rem 0.9rem",
          borderRadius: "var(--r-sm)",
          background: explain ? "var(--primary-tint)" : "transparent",
          border: `1px solid ${explain ? "var(--primary-tint-2)" : "var(--line)"}`,
          transition: "background var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease)",
        }}
      >
        {explain ? (
          <>
            <div
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-sm)",
                fontWeight: "var(--w-semibold)",
                color: "var(--primary)",
              }}
            >
              {explain.title}
            </div>
            <div
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-sm)",
                color: "var(--ink-2)",
                lineHeight: "var(--lh-snug)",
                marginTop: "0.2rem",
              }}
            >
              {explain.body}
            </div>
          </>
        ) : (
          <div
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: "var(--text-sm)",
              color: "var(--ink-3)",
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              height: "100%",
            }}
          >
            <span
              aria-hidden="true"
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                background: "var(--primary)",
                flexShrink: 0,
                opacity: 0.7,
              }}
            />
            {hint}
          </div>
        )}
      </div>
    </div>
  );
}

/** A labelled block in a network diagram. */
export function Block({
  x,
  y,
  w,
  h,
  label,
  sub,
  fill = "var(--surface)",
  stroke = "var(--line-2)",
  active,
  onMouseEnter,
  onMouseLeave,
  dim,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sub?: string;
  fill?: string;
  stroke?: string;
  active?: boolean;
  // Standard DOM handler names on purpose: the diagrams build one `on(part)`
  // helper and spread it onto BOTH <Block> and raw <g> elements. Custom names
  // here would silently no-op on the <g> spreads (React drops unknown props on
  // DOM elements with only a warning).
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  dim?: boolean;
}) {
  return (
    <g
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onFocus={onMouseEnter}
      onBlur={onMouseLeave}
      tabIndex={onMouseEnter ? 0 : undefined}
      style={{
        cursor: onMouseEnter ? "pointer" : "default",
        opacity: dim ? 0.32 : 1,
        transition: "opacity var(--dur-fast) var(--ease)",
        outline: "none",
      }}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={5}
        fill={fill}
        stroke={active ? "var(--primary)" : stroke}
        strokeWidth={active ? 2 : 1}
      />
      <text
        x={x + w / 2}
        y={y + (sub ? h / 2 - 3 : h / 2 + 4)}
        textAnchor="middle"
        style={{ font: "600 11px var(--font-sans)", fill: "var(--ink)" }}
      >
        {label}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 11}
          textAnchor="middle"
          style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-3)" }}
        >
          {sub}
        </text>
      )}
    </g>
  );
}

/** Arrow marker defs — include once per SVG that draws arrows. */
export function ArrowDefs() {
  return (
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--line-3)" />
      </marker>
      <marker
        id="arrow-active"
        viewBox="0 0 10 10"
        refX="9"
        refY="5"
        markerWidth="5"
        markerHeight="5"
        orient="auto-start-reverse"
      >
        <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--primary)" />
      </marker>
    </defs>
  );
}

export function Arrow({
  x1,
  y1,
  x2,
  y2,
  active,
  dim,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  active?: boolean;
  dim?: boolean;
}) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={active ? "var(--primary)" : "var(--line-3)"}
      strokeWidth={active ? 2 : 1.5}
      markerEnd={`url(#${active ? "arrow-active" : "arrow"})`}
      opacity={dim ? 0.3 : 1}
      style={{ transition: "opacity var(--dur-fast) var(--ease)" }}
    />
  );
}

export function Caption({ x, y, children }: { x: number; y: number; children: string }) {
  return (
    <text x={x} y={y} textAnchor="middle" style={{ font: "500 9px var(--font-mono)", fill: "var(--ink-4)" }}>
      {children}
    </text>
  );
}
