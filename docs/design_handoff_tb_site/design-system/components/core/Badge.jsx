import React from "react";

/**
 * Badge — small status pill. Carries result/status meaning by color + word.
 * tones: good | warn | bad | info | primary | neutral. styles: soft (tinted)
 * or solid (filled).
 */
export function Badge({ children, tone = "neutral", variant = "soft", style = {}, ...rest }) {
  const palette = {
    good: { fg: "var(--good)", bg: "var(--good-tint)", solid: "var(--good)" },
    warn: { fg: "var(--warn)", bg: "var(--warn-tint)", solid: "var(--warn)" },
    bad: { fg: "var(--bad)", bg: "var(--bad-tint)", solid: "var(--bad)" },
    info: { fg: "var(--info)", bg: "var(--info-tint)", solid: "var(--info)" },
    primary: { fg: "var(--primary)", bg: "var(--primary-tint)", solid: "var(--primary)" },
    neutral: { fg: "var(--ink-2)", bg: "var(--paper-2)", solid: "var(--ink)" },
  }[tone];

  const soft = {
    background: palette.bg,
    color: palette.fg,
    border: "1px solid transparent",
  };
  const solid = {
    background: palette.solid,
    color: tone === "neutral" ? "var(--paper)" : "#fff",
    border: "1px solid transparent",
  };

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.35rem",
        fontFamily: "var(--font-sans)",
        fontSize: "var(--text-label)",
        fontWeight: "var(--w-semibold)",
        textTransform: "uppercase",
        letterSpacing: "var(--ls-label)",
        padding: "0.28rem 0.6rem",
        borderRadius: "var(--r-pill)",
        lineHeight: 1,
        whiteSpace: "nowrap",
        ...(variant === "solid" ? solid : soft),
        ...style,
      }}
      {...rest}
    >
      {children}
    </span>
  );
}
