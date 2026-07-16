import { useNavigate } from "react-router-dom";
import { Button } from "@/design-system";
import { Icon } from "@/components/Icon";
import { ContentsList } from "@/components/Reading";
import { NAV_GROUPS } from "@/data/nav";

function Para({ children }: { children: React.ReactNode }) {
  return (
    <p
      style={{
        fontFamily: "var(--font-sans)",
        fontSize: "var(--text-lead)",
        color: "var(--ink-2)",
        lineHeight: "var(--lh-body)",
        margin: "0 0 var(--sp-4)",
        maxWidth: "68ch",
      }}
    >
      {children}
    </p>
  );
}

/** The three numbers that frame the whole project. */
function Headline() {
  const stats = [
    { v: "10.7M", l: "fell ill with TB in 2024", s: "1.23M of them died" },
    { v: "68.7%", l: "radiologist accuracy on this benchmark", s: "the honest human baseline" },
    { v: "0.745", l: "our detector's Active mAP50", s: "sealed test, multi-seed" },
  ];
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: "var(--sp-4)",
        margin: "var(--sp-6) 0",
      }}
    >
      {stats.map((s) => (
        <div
          key={s.v}
          style={{
            padding: "var(--sp-4)",
            border: "1px solid var(--line)",
            borderRadius: "var(--r)",
            background: "var(--surface)",
          }}
        >
          {/* Hero figures use the UI sans with proportional figures, not tabular. */}
          <div
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: "2rem",
              fontWeight: "var(--w-bold)",
              color: "var(--primary)",
              letterSpacing: "var(--ls-tight)",
              lineHeight: 1.1,
            }}
          >
            {s.v}
          </div>
          <div
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: "var(--text-sm)",
              color: "var(--ink-2)",
              marginTop: "0.3rem",
              lineHeight: "var(--lh-snug)",
            }}
          >
            {s.l}
          </div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--ink-3)", marginTop: "0.25rem" }}>
            {s.s}
          </div>
        </div>
      ))}
    </div>
  );
}

export function OverviewPage() {
  const navigate = useNavigate();
  const log = NAV_GROUPS[1].pages;

  return (
    <main style={{ maxWidth: "var(--content-max)", margin: "0 auto", padding: "0 var(--gutter) var(--sp-9)" }}>
      <header style={{ padding: "var(--sp-9) 0 var(--sp-5)" }}>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-label)",
            textTransform: "uppercase",
            letterSpacing: "var(--ls-label)",
            color: "var(--primary)",
            fontWeight: "var(--w-semibold)",
            marginBottom: "var(--sp-3)",
          }}
        >
          A living research paper
        </div>
        <h1
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "clamp(2.4rem, 5vw, 3.4rem)",
            fontWeight: "var(--w-semibold)",
            letterSpacing: "var(--ls-tight)",
            lineHeight: 1.08,
            margin: "0 0 var(--sp-5)",
            maxWidth: "18ch",
          }}
        >
          Detecting tuberculosis in chest X-rays
        </h1>
        <Para>
          Tuberculosis is curable, and it still killed 1.23 million people in 2024 — more than any other single
          infectious agent. Chest X-rays are the only screening tool cheap and fast enough to point at an entire
          population, but reading them needs a radiologist, and the countries with the most TB have the fewest.
        </Para>
        <Para>
          This project asks whether a deep-learning model can do that reading — not just score well on a benchmark, but
          hold up honestly under measurement. The site is both the write-up and the notebook: the literature review
          explains the problem and the methods from scratch, and the research log records every experiment, including
          the ones that failed and the conclusions we have had to retract.
        </Para>

        <Headline />

        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Button variant="primary" onClick={() => navigate("/about-tb")} iconRight={<Icon name="arrow-right" size="1em" />}>
            Start the review
          </Button>
          <Button variant="secondary" onClick={() => navigate("/results")}>
            Jump to results
          </Button>
        </div>
      </header>

      <section style={{ marginTop: "var(--sp-7)" }}>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "var(--text-h2)",
            fontWeight: "var(--w-semibold)",
            margin: "0 0 var(--sp-2)",
          }}
        >
          Contents
        </h2>
        <p style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-3)", margin: "0 0 var(--sp-4)" }}>
          The literature review, in reading order. Each section stands alone, but they build.
        </p>
        <ContentsList />
      </section>

      <section style={{ marginTop: "var(--sp-7)" }}>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "var(--text-h2)",
            fontWeight: "var(--w-semibold)",
            margin: "0 0 var(--sp-2)",
          }}
        >
          The research log
        </h2>
        <p style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-3)", margin: "0 0 var(--sp-4)" }}>
          The working record: what we ran, what it settled, and what it overturned.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "var(--sp-3)" }}>
          {log.map((p) => (
            <button
              key={p.path}
              onClick={() => navigate(p.path)}
              style={{
                all: "unset",
                boxSizing: "border-box",
                cursor: "pointer",
                display: "flex",
                gap: "0.75rem",
                padding: "var(--sp-4)",
                border: "1px solid var(--line)",
                borderRadius: "var(--r)",
                background: "var(--surface)",
                transition: "border-color var(--transition), box-shadow var(--transition)",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = "var(--primary-tint-2)";
                (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = "var(--line)";
                (e.currentTarget as HTMLElement).style.boxShadow = "none";
              }}
            >
              <Icon name={p.icon} size="1.1rem" style={{ color: "var(--primary)", marginTop: "0.15rem" }} />
              <span style={{ minWidth: 0 }}>
                <span
                  style={{
                    display: "block",
                    fontFamily: "var(--font-serif)",
                    fontSize: "1.02rem",
                    fontWeight: "var(--w-semibold)",
                    color: "var(--ink)",
                  }}
                >
                  {p.label}
                </span>
                <span
                  style={{
                    display: "block",
                    fontFamily: "var(--font-sans)",
                    fontSize: "var(--text-xs)",
                    color: "var(--ink-3)",
                    lineHeight: "var(--lh-snug)",
                    marginTop: "0.2rem",
                  }}
                >
                  {p.blurb}
                </span>
              </span>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}
