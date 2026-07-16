import React from "react";

interface StatCardProps {
  label: string;
  value: React.ReactNode;
  unit?: string;
  tone?: "default" | "good" | "warn" | "bad" | "primary";
  /** "sm" is for dense grids and for values that are text or carry a ± band. */
  size?: "sm" | "md";
  delta?: string;
  note?: string;
  style?: React.CSSProperties;
}

export function StatCard({ label, value, unit, tone = "default", size = "md", delta, note, style = {}, ...rest }: StatCardProps) {
  const valColor = {
    default: "var(--ink)",
    good:    "var(--good)",
    warn:    "var(--warn)",
    bad:     "var(--bad)",
    primary: "var(--primary)",
  }[tone];

  const sm = size === "sm";
  const deltaColor = delta && (delta.trim().startsWith("-") ? "var(--bad)" : "var(--good)");

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: "var(--r)", padding: sm ? "0.7rem 0.85rem" : "1rem 1.1rem", minWidth: 0, ...style }} {...rest}>
      <div style={{ fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", fontWeight: "var(--w-semibold)", color: "var(--ink-3)", marginBottom: sm ? "0.35rem" : "0.5rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={label}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", minWidth: 0 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: sm ? "1.05rem" : "1.6rem", fontWeight: "var(--w-semibold)", fontVariantNumeric: "tabular-nums", color: valColor, lineHeight: 1.15, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {value}
          {unit && <span style={{ fontSize: "0.85rem", color: "var(--ink-3)", marginLeft: "0.15rem", fontWeight: "var(--w-medium)" }}>{unit}</span>}
        </span>
        {delta && <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: "var(--w-semibold)", color: deltaColor || undefined }}>{delta}</span>}
      </div>
      {note && <div style={{ fontSize: "var(--text-xs)", color: "var(--ink-3)", marginTop: "0.4rem", lineHeight: "var(--lh-snug)" }}>{note}</div>}
    </div>
  );
}
