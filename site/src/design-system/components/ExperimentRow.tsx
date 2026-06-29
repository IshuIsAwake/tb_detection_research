import React from "react";

interface Metric { label: string; value: string; tone?: string; }

interface ExperimentRowProps {
  id: string;
  title: string;
  metrics?: Metric[];
  badge?: React.ReactNode;
  children?: React.ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onToggle?: (next: boolean) => void;
  style?: React.CSSProperties;
}

export function ExperimentRow({ id, title, metrics = [], badge, children, defaultOpen = false, open, onToggle, style = {} }: ExperimentRowProps) {
  const [internalOpen, setInternalOpen] = React.useState(defaultOpen);
  const isOpen = open !== undefined ? open : internalOpen;
  const toggle = () => (onToggle ? onToggle(!isOpen) : setInternalOpen((v) => !v));

  const metricColor = (t?: string) =>
    ({ good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)", default: "var(--ink)" }[t ?? "default"] ?? "var(--ink)");

  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: "var(--r)", background: "var(--surface)", boxShadow: isOpen ? "var(--shadow)" : "var(--shadow-sm)", overflow: "hidden", transition: "box-shadow var(--transition)", ...style }}>
      <button
        onClick={toggle}
        style={{ all: "unset", display: "flex", alignItems: "center", gap: "1.25rem", width: "100%", boxSizing: "border-box", padding: "1rem 1.15rem", cursor: "pointer", background: isOpen ? "var(--paper-2)" : "transparent", borderBottom: isOpen ? "1px solid var(--line)" : "1px solid transparent" }}
        onMouseEnter={(e) => { if (!isOpen) (e.currentTarget as HTMLElement).style.background = "var(--surface-2)"; }}
        onMouseLeave={(e) => { if (!isOpen) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
      >
        <span style={{ fontFamily: "var(--font-mono)", fontWeight: "var(--w-semibold)", fontSize: "var(--text-body)", color: "var(--primary)", flexShrink: 0, width: "3.5rem" }}>{id}</span>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-body)", color: "var(--ink)", flex: 1, minWidth: 0 }}>{title}</span>
        <span style={{ display: "flex", gap: "1.5rem", alignItems: "baseline", flexShrink: 0 }}>
          {metrics.map((m, i) => (
            <span key={i} style={{ textAlign: "right" }}>
              <span style={{ display: "block", fontSize: "0.62rem", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--ink-3)", fontWeight: "var(--w-semibold)" }}>{m.label}</span>
              <span style={{ fontFamily: "var(--font-mono)", fontVariantNumeric: "tabular-nums", fontWeight: "var(--w-semibold)", fontSize: "1.05rem", color: metricColor(m.tone) }}>{m.value}</span>
            </span>
          ))}
        </span>
        {badge && <span style={{ flexShrink: 0 }}>{badge}</span>}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--ink-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, transform: isOpen ? "rotate(180deg)" : "rotate(0)", transition: "transform var(--transition)" }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {isOpen && <div style={{ padding: "1.4rem 1.15rem", background: "var(--surface)" }}>{children}</div>}
    </div>
  );
}
