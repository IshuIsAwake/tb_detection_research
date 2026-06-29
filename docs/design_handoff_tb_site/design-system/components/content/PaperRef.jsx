import React from "react";

/**
 * PaperRef — a literature reading-list row. Title + authors/venue + a "why it
 * matters here" note, a mono arXiv/link chip, and a read-tracker checkbox.
 * `important` marks the must-read / non-negotiable citations.
 */
export function PaperRef({
  title,
  meta,
  link,
  linkLabel,
  note,
  important = false,
  read = false,
  onToggleRead,
  style = {},
}) {
  const [checked, setChecked] = React.useState(read);
  const isRead = onToggleRead ? read : checked;
  const toggle = () => (onToggleRead ? onToggleRead(!read) : setChecked((v) => !v));

  return (
    <div
      style={{
        display: "flex",
        gap: "0.9rem",
        padding: "1rem 1.1rem",
        background: "var(--surface)",
        border: "1px solid var(--line)",
        borderLeft: important ? "3px solid var(--primary)" : "1px solid var(--line)",
        borderRadius: "var(--r)",
        ...style,
      }}
    >
      <button
        onClick={toggle}
        aria-pressed={isRead}
        title={isRead ? "Read" : "Mark as read"}
        style={{
          all: "unset",
          cursor: "pointer",
          flexShrink: 0,
          width: "20px",
          height: "20px",
          marginTop: "0.15rem",
          borderRadius: "var(--r-sm)",
          border: `1.5px solid ${isRead ? "var(--good)" : "var(--line-3)"}`,
          background: isRead ? "var(--good)" : "transparent",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "all var(--transition)",
        }}
      >
        {isRead && (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="5 12 10 17 19 7" />
          </svg>
        )}
      </button>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.6rem", flexWrap: "wrap" }}>
          <span style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h3)", fontWeight: "var(--w-semibold)", color: "var(--ink)", lineHeight: 1.25 }}>
            {title}
          </span>
          {important && (
            <span style={{ fontFamily: "var(--font-sans)", fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "var(--ls-label)", fontWeight: "var(--w-semibold)", color: "var(--primary)" }}>
              Must read
            </span>
          )}
        </div>
        {meta && (
          <div style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-3)", marginTop: "0.2rem" }}>{meta}</div>
        )}
        {note && (
          <div style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-2)", marginTop: "0.45rem", lineHeight: "var(--lh-snug)" }}>{note}</div>
        )}
        {link && (
          <a
            href={link}
            target="_blank"
            rel="noreferrer"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.3rem",
              marginTop: "0.55rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.74rem",
              color: "var(--primary)",
              textDecoration: "none",
              borderBottom: "1px solid var(--primary-tint-2)",
              paddingBottom: "1px",
            }}
          >
            {linkLabel || link.replace(/^https?:\/\//, "")}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="7" y1="17" x2="17" y2="7" /><polyline points="8 7 17 7 17 16" />
            </svg>
          </a>
        )}
      </div>
    </div>
  );
}
