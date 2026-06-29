import React from "react";

/**
 * Tabs — segmented control. `items` is [{ id, label }]. Controlled via
 * `value`/`onChange`, or self-managed from `defaultValue`. Used for things
 * like confidence-threshold panels on the results page.
 */
export function Tabs({ items = [], value, defaultValue, onChange, style = {} }) {
  const [internal, setInternal] = React.useState(defaultValue ?? items[0]?.id);
  const active = value !== undefined ? value : internal;
  const select = (id) => (onChange ? onChange(id) : setInternal(id));

  return (
    <div
      role="tablist"
      style={{
        display: "inline-flex",
        gap: "0.2rem",
        padding: "0.25rem",
        background: "var(--paper-2)",
        border: "1px solid var(--line)",
        borderRadius: "var(--r)",
        ...style,
      }}
    >
      {items.map((it) => {
        const on = it.id === active;
        return (
          <button
            key={it.id}
            role="tab"
            aria-selected={on}
            onClick={() => select(it.id)}
            style={{
              all: "unset",
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-sm)",
              fontWeight: "var(--w-medium)",
              padding: "0.4rem 0.85rem",
              borderRadius: "var(--r-sm)",
              color: on ? "var(--on-primary)" : "var(--ink-2)",
              background: on ? "var(--primary)" : "transparent",
              transition: "background var(--transition), color var(--transition)",
              whiteSpace: "nowrap",
            }}
            onMouseEnter={(e) => { if (!on) e.currentTarget.style.background = "var(--surface)"; }}
            onMouseLeave={(e) => { if (!on) e.currentTarget.style.background = "transparent"; }}
          >
            {it.label}
          </button>
        );
      })}
    </div>
  );
}
