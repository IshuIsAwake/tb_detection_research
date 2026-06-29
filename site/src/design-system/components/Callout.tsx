import React from "react";

interface CalloutProps {
  kind?: "note" | "decision" | "landmine" | "retraction";
  title?: string;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

export function Callout({ kind = "note", title, children, style = {} }: CalloutProps) {
  const conf = {
    note:       { c: "var(--info)",  bg: "var(--info-tint)",  label: "Note",      icon: "info" },
    decision:   { c: "var(--good)",  bg: "var(--good-tint)",  label: "Decision",  icon: "check" },
    landmine:   { c: "var(--warn)",  bg: "var(--warn-tint)",  label: "Landmine",  icon: "alert" },
    retraction: { c: "var(--bad)",   bg: "var(--bad-tint)",   label: "Retracted", icon: "x" },
  }[kind];

  const icons: Record<string, React.ReactNode> = {
    info:  <><circle cx="12" cy="12" r="9" /><line x1="12" y1="11" x2="12" y2="16" /><line x1="12" y1="8" x2="12" y2="8" /></>,
    check: <><circle cx="12" cy="12" r="9" /><polyline points="8 12 11 15 16 9" /></>,
    alert: <><path d="M12 3 2 20h20L12 3Z" /><line x1="12" y1="10" x2="12" y2="14" /><line x1="12" y1="17" x2="12" y2="17" /></>,
    x:     <><circle cx="12" cy="12" r="9" /><line x1="9" y1="9" x2="15" y2="15" /><line x1="15" y1="9" x2="9" y2="15" /></>,
  };

  return (
    <div style={{ display: "flex", gap: "0.85rem", background: conf.bg, borderLeft: `3px solid ${conf.c}`, borderRadius: "0 var(--r) var(--r) 0", padding: "0.9rem 1.1rem", ...style }}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={conf.c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: "0.15rem" }}>
        {icons[conf.icon]}
      </svg>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", fontWeight: "var(--w-semibold)", color: conf.c, marginBottom: "0.3rem" }}>
          {title || conf.label}
        </div>
        <div style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-2)", lineHeight: "var(--lh-snug)" }}>
          {children}
        </div>
      </div>
    </div>
  );
}
