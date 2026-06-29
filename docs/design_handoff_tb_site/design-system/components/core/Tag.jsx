import React from "react";

/**
 * Tag — mono chip for configs, capabilities and identifiers
 * (e.g. "512 @ batch 16", "mosaic", "CC BY 4.0"). Lower-key than Badge:
 * monospace, sentence/lowercase, hairline border.
 */
export function Tag({ children, tone = "neutral", style = {}, ...rest }) {
  const tones = {
    neutral: { color: "var(--ink-2)", borderColor: "var(--line-2)", background: "var(--surface)" },
    primary: { color: "var(--primary)", borderColor: "var(--primary-tint-2)", background: "var(--primary-tint)" },
    muted: { color: "var(--ink-3)", borderColor: "var(--line)", background: "var(--paper-2)" },
  }[tone];

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.3rem",
        fontFamily: "var(--font-mono)",
        fontSize: "0.74rem",
        fontWeight: "var(--w-medium)",
        fontVariantNumeric: "tabular-nums",
        padding: "0.2rem 0.5rem",
        borderRadius: "var(--r-sm)",
        border: `1px solid ${tones.borderColor}`,
        background: tones.background,
        color: tones.color,
        lineHeight: 1.3,
        whiteSpace: "nowrap",
        ...style,
      }}
      {...rest}
    >
      {children}
    </span>
  );
}
