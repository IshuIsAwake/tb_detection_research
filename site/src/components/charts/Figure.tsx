import React from "react";
import { Icon } from "@/components/Icon";

export interface TableSpec {
  head: string[];
  rows: (string | number)[][];
}

/**
 * The frame every figure on the site shares: a numbered label, a title, the
 * figure itself, a caption, and a table-view twin.
 *
 * The table view is not optional decoration — it is the accessible equivalent of
 * the chart. A tooltip must never be the only way to read a value, so every
 * chart ships one.
 */
export function Figure({
  n,
  title,
  caption,
  source,
  table,
  children,
}: {
  n: string;
  title: string;
  caption?: React.ReactNode;
  source?: React.ReactNode;
  table?: TableSpec;
  children: React.ReactNode;
}) {
  const [showTable, setShowTable] = React.useState(false);

  return (
    <figure
      style={{
        margin: "var(--sp-5) 0",
        border: "1px solid var(--line)",
        borderRadius: "var(--r-lg)",
        background: "var(--surface)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: "1rem",
          padding: "0.9rem 1.15rem",
          borderBottom: "1px solid var(--line)",
          background: "var(--paper-2)",
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-label)",
              textTransform: "uppercase",
              letterSpacing: "var(--ls-label)",
              color: "var(--ink-3)",
              fontWeight: "var(--w-semibold)",
            }}
          >
            Figure {n}
          </div>
          <div
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: "1.02rem",
              fontWeight: "var(--w-semibold)",
              color: "var(--ink)",
              marginTop: "0.15rem",
              lineHeight: "var(--lh-snug)",
            }}
          >
            {title}
          </div>
        </div>
        {table && (
          <button
            onClick={() => setShowTable((v) => !v)}
            className="fig-toggle"
            aria-pressed={showTable}
            title={showTable ? "Show the chart" : "Show the numbers as a table"}
          >
            <Icon name={showTable ? "activity" : "database"} size="0.85rem" />
            <span>{showTable ? "Chart" : "Table"}</span>
          </button>
        )}
      </div>

      <div style={{ padding: "var(--sp-5) 1.15rem" }}>
        {showTable && table ? <DataTable {...table} /> : children}
      </div>

      {(caption || source) && (
        <figcaption
          style={{
            padding: "0.8rem 1.15rem",
            borderTop: "1px solid var(--line)",
            background: "var(--surface-2)",
            fontFamily: "var(--font-sans)",
            fontSize: "var(--text-sm)",
            color: "var(--ink-2)",
            lineHeight: "var(--lh-snug)",
          }}
        >
          {caption}
          {source && (
            <div style={{ marginTop: caption ? "0.4rem" : 0, color: "var(--ink-3)", fontSize: "var(--text-xs)" }}>
              {source}
            </div>
          )}
        </figcaption>
      )}
    </figure>
  );
}

export function DataTable({ head, rows }: TableSpec) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "320px" }}>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th
                key={h}
                style={{
                  textAlign: i === 0 ? "left" : "right",
                  whiteSpace: "nowrap",
                  padding: "0.5rem 0.7rem",
                  fontFamily: "var(--font-mono)",
                  fontSize: "var(--text-label)",
                  textTransform: "uppercase",
                  letterSpacing: "var(--ls-label)",
                  color: "var(--ink-3)",
                  fontWeight: "var(--w-semibold)",
                  borderBottom: "1px solid var(--line-2)",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td
                  key={ci}
                  style={{
                    textAlign: ci === 0 ? "left" : "right",
                    padding: "0.5rem 0.7rem",
                    fontFamily: ci === 0 ? "var(--font-sans)" : "var(--font-mono)",
                    fontSize: "var(--text-sm)",
                    color: ci === 0 ? "var(--ink)" : "var(--ink-2)",
                    fontVariantNumeric: "tabular-nums",
                    borderTop: "1px solid var(--line)",
                  }}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
