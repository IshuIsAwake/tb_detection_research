import React from "react";

interface StatCardProps {
  label: string;
  value: React.ReactNode;
  unit?: string;
  tone?: "default" | "good" | "warn" | "bad" | "primary";
  delta?: string;
  note?: string;
  style?: React.CSSProperties;
}

export function StatCard({ label, value, unit, tone = "default", delta, note, style = {}, ...rest }: StatCardProps) {
  const valColor = {
    default: "var(--ink)",
    good:    "var(--good)",
    warn:    "var(--warn)",
    bad:     "var(--bad)",
    primary: "var(--primary)",
  }[tone];

  const deltaColor = delta && (delta.trim().startsWith("-") ? "var(--bad)" : "var(--good)");

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: "var(--r)", padding: "1rem 1.1rem", ...style }} {...rest}>
      <div style={{ fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", fontWeight: "var(--w-semibold)", color: "var(--ink-3)", marginBottom: "0.5rem" }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "1.6rem", fontWeight: "var(--w-semibold)", fontVariantNumeric: "tabular-nums", color: valColor, lineHeight: 1.1 }}>
          {value}
          {unit && <span style={{ fontSize: "0.85rem", color: "var(--ink-3)", marginLeft: "0.15rem", fontWeight: "var(--w-medium)" }}>{unit}</span>}
        </span>
        {delta && <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: "var(--w-semibold)", color: deltaColor || undefined }}>{delta}</span>}
      </div>
      {note && <div style={{ fontSize: "var(--text-xs)", color: "var(--ink-3)", marginTop: "0.4rem" }}>{note}</div>}
    </div>
  );
}
