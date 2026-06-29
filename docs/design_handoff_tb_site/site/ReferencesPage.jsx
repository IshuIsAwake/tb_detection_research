const { PaperRef, Badge } = window.TBDetectionResearchDesignSystem_ccf532;

function ReferencesPage() {
  const [readState, setReadState] = React.useState(() => {
    const m = {};
    window.PAPER_GROUPS.forEach((g) => g.papers.forEach((p) => { m[p.title] = !!p.read; }));
    return m;
  });

  const total = Object.keys(readState).length;
  const done = Object.values(readState).filter(Boolean).length;

  return (
    <window.Page>
      <window.PageHeader
        eyebrow="Home / References"
        title="Literature review"
        lead="A reference list, not the review itself — titles, links, and a one-line note on why each matters here. The papers build on each other; read them in dependency order, not by date. Tick papers off as you read."
      />

      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "var(--sp-7)", padding: "1rem 1.2rem", background: "var(--primary-tint)", border: "1px solid var(--primary-tint-2)", borderRadius: "var(--r)" }}>
        <window.Icon name="book-open" color="var(--primary)" size="1.2rem" />
        <span style={{ fontSize: "var(--text-sm)", color: "var(--ink-2)" }}>
          Reading progress — <strong style={{ fontFamily: "var(--font-mono)", color: "var(--primary)" }}>{done} / {total}</strong> papers
        </span>
        <div style={{ flex: 1, height: "6px", background: "color-mix(in srgb, var(--primary) 18%, transparent)", borderRadius: "var(--r-pill)", overflow: "hidden", maxWidth: "260px" }}>
          <div style={{ width: (done / total * 100) + "%", height: "100%", background: "var(--primary)", transition: "width var(--dur-slow) var(--ease)" }} />
        </div>
      </div>

      {window.PAPER_GROUPS.map((g, gi) => (
        <section key={gi} style={{ marginBottom: "var(--sp-7)" }}>
          <window.SectionTitle kicker={`Section ${gi + 1}`}>{g.group}</window.SectionTitle>
          <p style={{ margin: "-0.75rem 0 var(--sp-4)", color: "var(--ink-2)", fontSize: "var(--text-sm)", maxWidth: "var(--prose-max)" }}>{g.blurb}</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
            {g.papers.map((p, i) => (
              <PaperRef
                key={i}
                {...p}
                read={readState[p.title]}
                onToggleRead={(next) => setReadState((s) => ({ ...s, [p.title]: next }))}
              />
            ))}
          </div>
        </section>
      ))}

      <window.PaperNote />
    </window.Page>
  );
}

function PaperNote() {
  const { Callout } = window.TBDetectionResearchDesignSystem_ccf532;
  return (
    <Callout kind="note" title="Citation note">
      YOLOv5 and YOLOv8 (the version this repo uses) have no peer-reviewed paper — they're Ultralytics
      software releases. Cite v1–v4 / v7 for the methodology and the Ultralytics repo for v8 itself.
    </Callout>
  );
}

window.PaperNote = PaperNote;
window.ReferencesPage = ReferencesPage;
