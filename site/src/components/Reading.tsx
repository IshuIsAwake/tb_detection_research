import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { Icon } from "@/components/Icon";
import { REVIEW_PAGES, reviewNeighbours, type NavPage } from "@/data/nav";

// ─── Citations ────────────────────────────────────────────────────────────────
//
// Scheme: PER-SECTION, DEDUPED. Each numbered section of the paper (i.e. each
// page) owns its own bibliography and its superscripts restart at 1. That keeps
// the numbers small and readable — ¹⁻⁴⁰ rather than ¹⁻¹⁷⁰ — and means moving a
// section never renumbers the rest of the site.
//
// The bibliography is a native <details>, which buys keyboard + screen-reader
// behaviour for free AND lets a <Cite> force it open by id without any context
// plumbing (the cite is rendered long before the bibliography mounts).

export interface Source {
  label: string;
  url: string;
}

const BIB_ID = "bibliography";
const entryId = (n: number) => `bib-${n}`;

/** Reveal a citation's entry: open the bibliography, scroll to it, flash it. */
function revealSource(n: number) {
  const details = document.getElementById(BIB_ID) as HTMLDetailsElement | null;
  if (details && !details.open) details.open = true;
  const el = document.getElementById(entryId(n));
  if (!el) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  requestAnimationFrame(() => {
    el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
    el.classList.remove("bib-flash");
    // Force a reflow so the animation restarts on repeat clicks.
    void el.offsetWidth;
    el.classList.add("bib-flash");
  });
}

/**
 * Bind a <Cite> to a page's source list so the numbers stay in sync.
 *
 *   const SOURCES: Source[] = [...];
 *   const Cite = makeCite(SOURCES);
 *   …<Cite n={1} />  or  <Cite n={[1, 4]} />
 */
export function makeCite(sources: Source[]) {
  if (import.meta.env.DEV) {
    const seen = new Map<string, number>();
    sources.forEach((s, i) => {
      const prev = seen.get(s.url);
      if (prev !== undefined) {
        console.warn(
          `[cite] duplicate source URL at [${prev + 1}] and [${i + 1}] — dedupe the list: ${s.url}`,
        );
      }
      seen.set(s.url, i);
    });
  }

  return function Cite({ n }: { n: number | number[] }) {
    const nums = Array.isArray(n) ? n : [n];
    return (
      <sup className="cite">
        {nums.map((num, i) => {
          const s = sources[num - 1];
          if (import.meta.env.DEV && !s) {
            console.warn(`[cite] no source [${num}] — the list has ${sources.length} entries`);
          }
          return (
            <React.Fragment key={num}>
              {i > 0 && <span style={{ color: "var(--ink-4)" }}>,</span>}
              <button
                type="button"
                className="cite-link"
                title={s?.label ?? `Source ${num}`}
                aria-label={`Citation ${num}: ${s?.label ?? ""}`}
                onClick={() => revealSource(num)}
              >
                {num}
              </button>
            </React.Fragment>
          );
        })}
      </sup>
    );
  };
}

/**
 * The section's works-cited list, collapsed by default. Long reference lists are
 * the point of the collapse: a 40-entry block at the foot of every section would
 * bury the prose.
 */
export function Bibliography({
  sources,
  title = "References",
  defaultOpen = false,
}: {
  sources: Source[];
  title?: string;
  defaultOpen?: boolean;
}) {
  return (
    <details id={BIB_ID} open={defaultOpen} className="bib">
      <summary className="bib-summary">
        <span className="bib-chev" aria-hidden="true">
          <Icon name="arrow-right" size="0.85rem" />
        </span>
        <span
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "var(--text-h3)",
            fontWeight: "var(--w-semibold)",
            color: "var(--ink)",
          }}
        >
          {title}
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.72rem",
            color: "var(--ink-3)",
            border: "1px solid var(--line-2)",
            borderRadius: "var(--r-pill)",
            padding: "0.05rem 0.45rem",
          }}
        >
          {sources.length}
        </span>
      </summary>
      <ol className="bib-list">
        {sources.map((s, i) => (
          <li key={s.url + i} id={entryId(i + 1)} className="bib-item">
            <span className="bib-n">{i + 1}</span>
            <a href={s.url} target="_blank" rel="noreferrer" className="bib-a">
              {s.label}
              <Icon name="arrow-up-right" size="0.72rem" style={{ opacity: 0.5, marginLeft: "0.25rem" }} />
            </a>
          </li>
        ))}
      </ol>
    </details>
  );
}

// ─── Section heading ──────────────────────────────────────────────────────────
// `data-section` is what the sidebar's outline discovers — see Sidebar.tsx. The
// prose is therefore the single source of truth for the outline.

export function ReadSection({
  id,
  n,
  title,
  lead,
  children,
}: {
  id: string;
  n: number;
  title: string;
  lead?: string;
  children: React.ReactNode;
}) {
  const num = String(n).padStart(2, "0");
  return (
    <section
      id={id}
      data-section={title}
      data-section-n={num}
      style={{ marginTop: "var(--sp-8)", scrollMarginTop: "72px" }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.7rem", marginBottom: "var(--sp-4)" }}>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.85rem",
            fontWeight: "var(--w-semibold)",
            color: "var(--primary)",
            border: "1px solid var(--primary-tint-2)",
            background: "var(--primary-tint)",
            borderRadius: "var(--r-sm)",
            padding: "0.15rem 0.5rem",
          }}
        >
          {num}
        </span>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "var(--text-h2)",
            fontWeight: "var(--w-semibold)",
            margin: 0,
            lineHeight: "var(--lh-snug)",
          }}
        >
          {title}
        </h2>
      </div>
      {lead && (
        <p
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "var(--text-lead)",
            color: "var(--ink-2)",
            lineHeight: "var(--lh-body)",
            maxWidth: "var(--prose-max)",
            margin: "0 0 var(--sp-5)",
          }}
        >
          {lead}
        </p>
      )}
      {children}
    </section>
  );
}

// ─── Page scaffolding for a review section ────────────────────────────────────

export function ReviewPage({
  path,
  children,
}: {
  path: string;
  children: React.ReactNode;
}) {
  const page = REVIEW_PAGES.find((p) => p.path === path);
  return (
    <main
      style={{
        maxWidth: "var(--content-max)",
        margin: "0 auto",
        padding: "var(--sp-7) var(--gutter) var(--sp-8)",
      }}
    >
      {children}
      {page && <Pager path={path} />}
    </main>
  );
}

export function ReviewHeader({
  n,
  title,
  lead,
}: {
  n: number;
  title: string;
  lead?: React.ReactNode;
}) {
  return (
    <header style={{ marginBottom: "var(--sp-6)" }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "var(--text-label)",
          textTransform: "uppercase",
          letterSpacing: "var(--ls-label)",
          color: "var(--primary)",
          fontWeight: "var(--w-semibold)",
        }}
      >
        Section {n} of {REVIEW_PAGES.length} · Literature review
      </div>
      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "var(--text-display)",
          fontWeight: "var(--w-semibold)",
          letterSpacing: "var(--ls-tight)",
          lineHeight: "var(--lh-tight)",
          margin: "0.5rem 0 0",
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
            margin: "var(--sp-4) 0 0",
          }}
        >
          {lead}
        </p>
      )}
    </header>
  );
}

// ─── Prev / next pager ────────────────────────────────────────────────────────

function PagerCard({ page, dir }: { page: NavPage; dir: "prev" | "next" }) {
  const navigate = useNavigate();
  const [hover, setHover] = React.useState(false);
  const next = dir === "next";
  return (
    <button
      onClick={() => navigate(page.path)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        all: "unset",
        boxSizing: "border-box",
        cursor: "pointer",
        flex: "1 1 240px",
        display: "flex",
        alignItems: "center",
        justifyContent: next ? "space-between" : "flex-start",
        flexDirection: next ? "row" : "row-reverse",
        gap: "1rem",
        padding: "var(--sp-5)",
        border: `1px solid ${hover ? "var(--primary-tint-2)" : "var(--line)"}`,
        background: "var(--surface)",
        borderRadius: "var(--r)",
        boxShadow: hover ? "var(--shadow)" : "none",
        transition: "border-color var(--transition), box-shadow var(--transition)",
        textAlign: next ? "left" : "right",
      }}
    >
      <span style={{ minWidth: 0 }}>
        <span
          style={{
            display: "block",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-label)",
            textTransform: "uppercase",
            letterSpacing: "var(--ls-label)",
            color: "var(--ink-3)",
            marginBottom: "0.25rem",
          }}
        >
          {next ? "Next" : "Previous"} · Section {page.n}
        </span>
        <span
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "var(--text-h3)",
            fontWeight: "var(--w-semibold)",
            color: "var(--ink)",
          }}
        >
          {page.label}
        </span>
      </span>
      <span
        style={{
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: "38px",
          height: "38px",
          borderRadius: "var(--r-pill)",
          background: hover ? "var(--primary)" : "var(--primary-tint)",
          color: hover ? "var(--on-primary)" : "var(--primary)",
          transition: "background var(--transition), color var(--transition)",
        }}
      >
        <Icon name={next ? "arrow-right" : "arrow-left"} size="1.05rem" />
      </span>
    </button>
  );
}

export function Pager({ path }: { path: string }) {
  const { prev, next } = reviewNeighbours(path);
  return (
    <nav
      aria-label="Section navigation"
      style={{ display: "flex", gap: "var(--sp-4)", marginTop: "var(--sp-8)", flexWrap: "wrap" }}
    >
      {prev ? <PagerCard page={prev} dir="prev" /> : <span style={{ flex: "1 1 240px" }} />}
      {next ? (
        <PagerCard page={next} dir="next" />
      ) : (
        <div
          style={{
            flex: "1 1 240px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "var(--sp-5)",
            border: "1px dashed var(--line-2)",
            borderRadius: "var(--r)",
            color: "var(--ink-3)",
            fontSize: "var(--text-sm)",
            textAlign: "center",
          }}
        >
          End of the literature review — the research log picks up from here.
        </div>
      )}
    </nav>
  );
}

// ─── Contents list (home page) ────────────────────────────────────────────────

export function ContentsList() {
  return (
    <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: "var(--sp-2)" }}>
      {REVIEW_PAGES.map((r) => (
        <li key={r.path}>
          <Link to={r.path} className="contents-row">
            <span className="contents-n">{String(r.n).padStart(2, "0")}</span>
            <span style={{ minWidth: 0 }}>
              <span
                style={{
                  display: "block",
                  fontFamily: "var(--font-serif)",
                  fontSize: "1.05rem",
                  fontWeight: "var(--w-semibold)",
                  color: "var(--ink)",
                }}
              >
                {r.label}
              </span>
              <span
                style={{
                  display: "block",
                  fontFamily: "var(--font-sans)",
                  fontSize: "var(--text-sm)",
                  color: "var(--ink-3)",
                  lineHeight: "var(--lh-snug)",
                  marginTop: "0.2rem",
                }}
              >
                {r.blurb}
              </span>
            </span>
            <Icon name="arrow-right" size="1rem" style={{ opacity: 0.4, flexShrink: 0 }} />
          </Link>
        </li>
      ))}
    </ol>
  );
}

// ─── Collapsible / accordion ──────────────────────────────────────────────────

export function Collapsible({
  title,
  subtitle,
  defaultOpen = false,
  children,
}: {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div
      style={{
        border: "1px solid var(--line)",
        borderRadius: "var(--r)",
        background: "var(--surface)",
        overflow: "hidden",
      }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          all: "unset",
          boxSizing: "border-box",
          cursor: "pointer",
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "1rem",
          padding: "1rem 1.15rem",
          background: open ? "var(--paper-2)" : "transparent",
          borderBottom: open ? "1px solid var(--line)" : "1px solid transparent",
          transition: "background var(--transition)",
        }}
      >
        <span style={{ minWidth: 0 }}>
          <span
            style={{
              display: "block",
              fontFamily: "var(--font-serif)",
              fontSize: "var(--text-h3)",
              fontWeight: "var(--w-semibold)",
              color: "var(--ink)",
            }}
          >
            {title}
          </span>
          {subtitle && (
            <span
              style={{
                display: "block",
                fontFamily: "var(--font-sans)",
                fontSize: "var(--text-xs)",
                color: "var(--ink-3)",
                marginTop: "0.2rem",
              }}
            >
              {subtitle}
            </span>
          )}
        </span>
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--ink-3)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{
            flexShrink: 0,
            transform: open ? "rotate(180deg)" : "rotate(0)",
            transition: "transform var(--transition)",
          }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && <div style={{ padding: "var(--sp-5) 1.15rem" }}>{children}</div>}
    </div>
  );
}

// ─── Content-card primitives ──────────────────────────────────────────────────

export function InfoCard({
  title,
  tone = "neutral",
  children,
  style = {},
}: {
  title?: React.ReactNode;
  tone?: "neutral" | "danger" | "muted" | "primary";
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  const tones = {
    neutral: { border: "var(--line)", bg: "var(--surface)" },
    danger: { border: "var(--bad)", bg: "var(--bad-tint)" },
    muted: { border: "var(--line-2)", bg: "var(--paper-2)" },
    primary: { border: "var(--primary-tint-2)", bg: "var(--primary-tint)" },
  }[tone];
  return (
    <div
      style={{
        border: `1px solid ${tones.border}`,
        background: tones.bg,
        borderRadius: "var(--r-lg)",
        padding: "var(--sp-5)",
        ...style,
      }}
    >
      {title && (
        <h3
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "var(--text-h3)",
            fontWeight: "var(--w-semibold)",
            margin: "0 0 var(--sp-3)",
            color: "var(--ink)",
          }}
        >
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

/** Responsive card grid — collapses to one column on narrow viewports. */
export function CardGrid({
  cols = 2,
  children,
  style = {},
}: {
  cols?: number;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div className={`card-grid card-grid-${cols}`} style={style}>
      {children}
    </div>
  );
}

export function Bullets({ items, style = {} }: { items: React.ReactNode[]; style?: React.CSSProperties }) {
  return (
    <ul style={{ margin: 0, paddingLeft: "1.15rem", display: "grid", gap: "0.5rem", ...style }}>
      {items.map((it, i) => (
        <li
          key={i}
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: "var(--text-sm)",
            color: "var(--ink-2)",
            lineHeight: "var(--lh-snug)",
          }}
        >
          {it}
        </li>
      ))}
    </ul>
  );
}

export function Prose({ children, style = {} }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <p
      style={{
        fontFamily: "var(--font-sans)",
        fontSize: "var(--text-body)",
        color: "var(--ink-2)",
        lineHeight: "var(--lh-body)",
        maxWidth: "var(--prose-max)",
        margin: "0 0 var(--sp-4)",
        ...style,
      }}
    >
      {children}
    </p>
  );
}

/** A pull-quote / key-idea aside for the long reading pages. */
export function KeyIdea({ children }: { children: React.ReactNode }) {
  return (
    <aside
      style={{
        display: "flex",
        gap: "var(--sp-4)",
        margin: "var(--sp-5) 0",
        padding: "var(--sp-5)",
        borderLeft: "3px solid var(--primary)",
        background: "var(--primary-tint)",
        borderRadius: "0 var(--r) var(--r) 0",
        maxWidth: "var(--prose-max)",
      }}
    >
      <Icon name="quote" size="1.1rem" style={{ color: "var(--primary)", flexShrink: 0, marginTop: "0.2rem" }} />
      <div
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "1.05rem",
          fontStyle: "italic",
          color: "var(--ink)",
          lineHeight: "var(--lh-snug)",
        }}
      >
        {children}
      </div>
    </aside>
  );
}
