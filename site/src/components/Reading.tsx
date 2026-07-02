import React from "react";
import { useNavigate, Link } from "react-router-dom";
import { Icon } from "@/components/Shell";

// ─── Reading sequence ─────────────────────────────────────────────────────────
// The three "reads" that live inside the Home flow, in intended reading order.

export interface Read {
  path: string;
  n: number;
  label: string;
  blurb: string;
}

export const READS: Read[] = [
  { path: "/about-tb",           n: 1, label: "About TB",                 blurb: "What TB is, its two forms, key properties, and global impact." },
  { path: "/how-tb-detected",    n: 2, label: "How is TB Detected",       blurb: "Why we screen rather than treat everyone, and how latent vs active TB are found." },
  { path: "/active-tb-detection", n: 3, label: "How is Active TB Detected", blurb: "The four frontline diagnostic methods, compared." },
];

// ─── TOC stepper (top of every reading page) ──────────────────────────────────

export function ReadingNav({ current }: { current: number }) {
  const navigate = useNavigate();
  return (
    <nav aria-label="Reading sections" style={{ display: "flex", flexWrap: "wrap", alignItems: "stretch", gap: "0.5rem", border: "1px solid var(--line)", background: "var(--surface)", borderRadius: "var(--r)", padding: "0.5rem", marginBottom: "var(--sp-7)" }}>
      {READS.map((r) => {
        const active = r.n === current;
        return (
          <button
            key={r.path}
            onClick={() => !active && navigate(r.path)}
            aria-current={active ? "step" : undefined}
            style={{
              all: "unset", boxSizing: "border-box", flex: "1 1 200px", minWidth: 0,
              cursor: active ? "default" : "pointer",
              display: "flex", alignItems: "center", gap: "0.6rem",
              padding: "0.6rem 0.75rem", borderRadius: "var(--r-sm)",
              background: active ? "var(--primary-tint)" : "transparent",
              border: active ? "1px solid var(--primary-tint-2)" : "1px solid transparent",
              transition: "background var(--transition)",
            }}
            onMouseEnter={(e) => { if (!active) (e.currentTarget as HTMLElement).style.background = "var(--paper-2)"; }}
            onMouseLeave={(e) => { if (!active) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
          >
            <span style={{
              flexShrink: 0, width: "26px", height: "26px", borderRadius: "var(--r-pill)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: "var(--w-semibold)",
              background: active ? "var(--primary)" : "transparent",
              color: active ? "var(--on-primary)" : "var(--ink-3)",
              border: active ? "1px solid var(--primary)" : "1px solid var(--line-2)",
            }}>{r.n}</span>
            <span style={{ minWidth: 0, fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", fontWeight: "var(--w-semibold)", color: active ? "var(--primary)" : "var(--ink-2)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {r.label}
            </span>
          </button>
        );
      })}
    </nav>
  );
}

// ─── Next-section footer (bottom of every reading page) ───────────────────────

export function NextSection({ current }: { current: number }) {
  const navigate = useNavigate();
  const next = READS.find((r) => r.n === current + 1);

  if (!next) {
    return (
      <div style={{ marginTop: "var(--sp-8)", padding: "var(--sp-5)", border: "1px dashed var(--line-2)", borderRadius: "var(--r)", textAlign: "center", color: "var(--ink-3)", fontSize: "var(--text-sm)" }}>
        You've reached the end of the current reads. More sections are planned.
      </div>
    );
  }

  return (
    <button
      onClick={() => navigate(next.path)}
      style={{ all: "unset", boxSizing: "border-box", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", width: "100%", marginTop: "var(--sp-8)", padding: "var(--sp-5)", border: "1px solid var(--line)", background: "var(--surface)", borderRadius: "var(--r)", transition: "border-color var(--transition), box-shadow var(--transition)" }}
      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "var(--primary-tint-2)"; (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow)"; }}
      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "var(--line)"; (e.currentTarget as HTMLElement).style.boxShadow = "none"; }}
    >
      <span>
        <span style={{ display: "block", fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--ink-3)", marginBottom: "0.25rem" }}>
          Next &middot; {next.n} of {READS.length}
        </span>
        <span style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h3)", fontWeight: "var(--w-semibold)", color: "var(--ink)" }}>{next.label}</span>
      </span>
      <span style={{ flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", width: "40px", height: "40px", borderRadius: "var(--r-pill)", background: "var(--primary)", color: "var(--on-primary)" }}>
        <Icon name="arrow-right" size="1.1rem" />
      </span>
    </button>
  );
}

// ─── Contents dropdown (Home page) ────────────────────────────────────────────

export function ContentsDropdown() {
  const [open, setOpen] = React.useState(false);
  return (
    <div style={{ maxWidth: "560px", border: "1px solid var(--line)", borderRadius: "var(--r)", background: "var(--surface)", overflow: "hidden" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{ all: "unset", boxSizing: "border-box", cursor: "pointer", width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", padding: "0.8rem 1rem", background: open ? "var(--paper-2)" : "transparent", transition: "background var(--transition)" }}
        onMouseEnter={(e) => { if (!open) (e.currentTarget as HTMLElement).style.background = "var(--surface-2)"; }}
        onMouseLeave={(e) => { if (!open) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
      >
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", fontWeight: "var(--w-semibold)", color: "var(--ink-2)" }}>
          Jump to a section
        </span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--ink-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? "rotate(180deg)" : "rotate(0)", transition: "transform var(--transition)" }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && (
        <ol style={{ listStyle: "none", margin: 0, padding: "0.4rem", borderTop: "1px solid var(--line)" }}>
          {READS.map((r) => (
            <li key={r.path}>
              <Link
                to={r.path}
                style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem", padding: "0.6rem 0.7rem", borderRadius: "var(--r-sm)", textDecoration: "none", transition: "background var(--transition)" }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--paper-2)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <span style={{ flexShrink: 0, fontFamily: "var(--font-mono)", fontSize: "0.78rem", fontWeight: "var(--w-semibold)", color: "var(--primary)", marginTop: "0.1rem" }}>{r.n}</span>
                <span style={{ minWidth: 0 }}>
                  <span style={{ display: "block", fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", fontWeight: "var(--w-semibold)", color: "var(--ink)" }}>{r.label}</span>
                  <span style={{ display: "block", fontFamily: "var(--font-sans)", fontSize: "var(--text-xs)", color: "var(--ink-3)", lineHeight: "var(--lh-snug)", marginTop: "0.15rem" }}>{r.blurb}</span>
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

// ─── Citations ────────────────────────────────────────────────────────────────

export interface Source { label: string; url: string; }

// Bind a Cite component + Sources list to a page's source list so numbers stay
// in sync. Usage: const Cite = makeCite(SOURCES); … <Cite n={1} /> or <Cite n={[1,2]} />
export function makeCite(sources: Source[]) {
  return function Cite({ n }: { n: number | number[] }) {
    const nums = Array.isArray(n) ? n : [n];
    return (
      <sup style={{ fontFamily: "var(--font-mono)", fontSize: "0.62em", fontWeight: "var(--w-semibold)", lineHeight: 0, whiteSpace: "nowrap" }}>
        {nums.map((num, i) => {
          const s = sources[num - 1];
          return (
            <React.Fragment key={num}>
              {i > 0 && <span style={{ color: "var(--ink-4)" }}>,</span>}
              <a href={s?.url} target="_blank" rel="noreferrer" title={s?.label}
                style={{ color: "var(--primary)", textDecoration: "none", padding: "0 0.05em" }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.textDecoration = "underline"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.textDecoration = "none"; }}
              >[{num}]</a>
            </React.Fragment>
          );
        })}
      </sup>
    );
  };
}

export function SourceList({ sources }: { sources: Source[] }) {
  return (
    <section style={{ marginTop: "var(--sp-7)", paddingTop: "var(--sp-5)", borderTop: "1px solid var(--line)" }}>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", fontWeight: "var(--w-semibold)", color: "var(--ink-3)", marginBottom: "var(--sp-3)" }}>Sources</div>
      <ol style={{ margin: 0, paddingLeft: "1.4rem", display: "grid", gap: "0.4rem" }}>
        {sources.map((s, i) => (
          <li key={i} style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-2)", lineHeight: "var(--lh-snug)" }}>
            <a href={s.url} target="_blank" rel="noreferrer" style={{ color: "var(--primary)", textDecoration: "none" }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.textDecoration = "underline"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.textDecoration = "none"; }}
            >{s.label}</a>
          </li>
        ))}
      </ol>
    </section>
  );
}

// ─── Collapsible / accordion ──────────────────────────────────────────────────

export function Collapsible({ title, subtitle, defaultOpen = false, children }: { title: string; subtitle?: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: "var(--r)", background: "var(--surface)", overflow: "hidden" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{ all: "unset", boxSizing: "border-box", cursor: "pointer", width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", padding: "1rem 1.15rem", background: open ? "var(--paper-2)" : "transparent", borderBottom: open ? "1px solid var(--line)" : "1px solid transparent", transition: "background var(--transition)" }}
        onMouseEnter={(e) => { if (!open) (e.currentTarget as HTMLElement).style.background = "var(--surface-2)"; }}
        onMouseLeave={(e) => { if (!open) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
      >
        <span style={{ minWidth: 0 }}>
          <span style={{ display: "block", fontFamily: "var(--font-serif)", fontSize: "var(--text-h3)", fontWeight: "var(--w-semibold)", color: "var(--ink)" }}>{title}</span>
          {subtitle && <span style={{ display: "block", fontFamily: "var(--font-sans)", fontSize: "var(--text-xs)", color: "var(--ink-3)", marginTop: "0.2rem" }}>{subtitle}</span>}
        </span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--ink-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, transform: open ? "rotate(180deg)" : "rotate(0)", transition: "transform var(--transition)" }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && <div style={{ padding: "var(--sp-5) 1.15rem" }}>{children}</div>}
    </div>
  );
}

// ─── Content-card primitives ──────────────────────────────────────────────────

export function InfoCard({ title, tone = "neutral", children, style = {} }: { title?: React.ReactNode; tone?: "neutral" | "danger" | "muted" | "primary"; children: React.ReactNode; style?: React.CSSProperties }) {
  const tones = {
    neutral: { border: "var(--line)",   bg: "var(--surface)" },
    danger:  { border: "var(--bad)",    bg: "var(--bad-tint)" },
    muted:   { border: "var(--line-2)", bg: "var(--paper-2)" },
    primary: { border: "var(--primary-tint-2)", bg: "var(--primary-tint)" },
  }[tone];
  return (
    <div style={{ border: `1px solid ${tones.border}`, background: tones.bg, borderRadius: "var(--r-lg)", padding: "var(--sp-5)", ...style }}>
      {title && <h3 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h3)", fontWeight: "var(--w-semibold)", margin: "0 0 var(--sp-3)", color: "var(--ink)" }}>{title}</h3>}
      {children}
    </div>
  );
}

export function CardGrid({ cols = 2, children, style = {} }: { cols?: number; children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: "var(--sp-4)", ...style }}>
      {children}
    </div>
  );
}

// Bulleted list with consistent spacing, used across reading pages.
export function Bullets({ items, style = {} }: { items: React.ReactNode[]; style?: React.CSSProperties }) {
  return (
    <ul style={{ margin: 0, paddingLeft: "1.15rem", display: "grid", gap: "0.5rem", ...style }}>
      {items.map((it, i) => (
        <li key={i} style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-2)", lineHeight: "var(--lh-snug)" }}>{it}</li>
      ))}
    </ul>
  );
}

// Body paragraph with reading-page defaults.
export function Prose({ children, style = {} }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <p style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-body)", color: "var(--ink-2)", lineHeight: "var(--lh-body)", maxWidth: "var(--prose-max)", margin: "0 0 var(--sp-4)", ...style }}>
      {children}
    </p>
  );
}

// Numbered section heading used down the length of a reading page.
export function ReadSection({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginTop: "var(--sp-8)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.7rem", marginBottom: "var(--sp-4)" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", fontWeight: "var(--w-semibold)", color: "var(--primary)", border: "1px solid var(--primary-tint-2)", background: "var(--primary-tint)", borderRadius: "var(--r-sm)", padding: "0.15rem 0.5rem" }}>{String(n).padStart(2, "0")}</span>
        <h2 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h2)", fontWeight: "var(--w-semibold)", margin: 0, lineHeight: "var(--lh-snug)" }}>{title}</h2>
      </div>
      {children}
    </section>
  );
}
