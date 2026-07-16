import React from "react";
import { Button, Badge } from "@/design-system";
import { Icon } from "@/components/Icon";

// The top bar is gone — navigation lives in components/Sidebar.tsx. What remains
// here are the layout primitives every page composes with.

export function Eyebrow({
  children,
  color = "var(--primary)",
}: {
  children: React.ReactNode;
  color?: string;
}) {
  return (
    <div
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: "var(--text-label)",
        textTransform: "uppercase",
        letterSpacing: "var(--ls-label)",
        fontWeight: "var(--w-semibold)",
        color,
      }}
    >
      {children}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  lead,
}: {
  eyebrow?: string;
  title: string;
  lead?: string;
}) {
  return (
    <header
      style={{
        padding: "var(--sp-7) 0 var(--sp-6)",
        borderBottom: "1px solid var(--line)",
        marginBottom: "var(--sp-7)",
      }}
    >
      {eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}
      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "var(--text-display)",
          fontWeight: "var(--w-semibold)",
          letterSpacing: "var(--ls-tight)",
          lineHeight: "var(--lh-tight)",
          margin: eyebrow ? "0.6rem 0 0" : 0,
        }}
      >
        {title}
      </h1>
      {lead && (
        <p
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "var(--text-lead)",
            color: "var(--ink-2)",
            lineHeight: "var(--lh-body)",
            maxWidth: "var(--prose-max)",
            margin: "1rem 0 0",
          }}
        >
          {lead}
        </p>
      )}
    </header>
  );
}

export function SectionTitle({
  kicker,
  children,
  right,
}: {
  kicker?: string;
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        gap: "1rem",
        // Wrap rather than crushing the heading to one word per line when a
        // `right` slot (a legend, a control) shares the row on a narrow screen.
        flexWrap: "wrap",
        marginBottom: "var(--sp-5)",
      }}
    >
      <div style={{ minWidth: 0 }}>
        {kicker && <Eyebrow color="var(--ink-3)">{kicker}</Eyebrow>}
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "var(--text-h2)",
            fontWeight: "var(--w-semibold)",
            margin: kicker ? "0.4rem 0 0" : 0,
            lineHeight: "var(--lh-snug)",
          }}
        >
          {children}
        </h2>
      </div>
      {right}
    </div>
  );
}

export function Page({ children }: { children: React.ReactNode }) {
  return (
    <main
      style={{
        maxWidth: "var(--content-max)",
        margin: "0 auto",
        padding: "0 var(--gutter) var(--sp-9)",
      }}
    >
      {children}
    </main>
  );
}

// Re-export so existing pages can keep doing: import { Icon, Page, … } from "@/components/Shell"
export { Icon, Button, Badge };
