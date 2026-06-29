const { StatCard, Callout, Badge, Tag, Button } = window.TBDetectionResearchDesignSystem_ccf532;

function CapMark({ value }) {
  const map = {
    yes: { icon: "check", color: "var(--good)", title: "Yes" },
    partial: { icon: "minus", color: "var(--warn)", title: "Partial" },
    no: { icon: "x", color: "var(--ink-4)", title: "No" },
  };
  const c = map[value] || map.no;
  return (
    <span title={c.title} style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: "22px", height: "22px", borderRadius: "var(--r-pill)", background: value === "no" ? "transparent" : "color-mix(in srgb, " + c.color + " 12%, transparent)" }}>
      <window.Icon name={c.icon} color={c.color} size="0.95rem" />
    </span>
  );
}

function DatasetRow({ ds, open, onToggle }) {
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: "var(--r)", background: "var(--surface)", boxShadow: open ? "var(--shadow)" : "var(--shadow-sm)", overflow: "hidden", transition: "box-shadow var(--transition)" }}>
      <button onClick={onToggle}
        style={{ all: "unset", boxSizing: "border-box", cursor: "pointer", width: "100%", display: "grid", gridTemplateColumns: "1.5fr 0.95fr 0.5fr 0.5fr 0.7fr 0.8fr auto", alignItems: "center", gap: "1rem", padding: "1rem 1.15rem", background: open ? "var(--paper-2)" : "transparent", borderBottom: open ? "1px solid var(--line)" : "1px solid transparent" }}
        onMouseEnter={(e) => { if (!open) e.currentTarget.style.background = "var(--surface-2)"; }}
        onMouseLeave={(e) => { if (!open) e.currentTarget.style.background = "transparent"; }}
      >
        <div style={{ minWidth: 0 }}>
          <div style={{ fontFamily: "var(--font-sans)", fontWeight: "var(--w-semibold)", fontSize: "var(--text-body)", color: "var(--ink)" }}>{ds.name}</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-3)", marginTop: "0.15rem" }}>{ds.host}</div>
        </div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--ink-2)", fontVariantNumeric: "tabular-nums" }}>{ds.tb}</div>
        <div style={{ textAlign: "center" }}><CapMark value={ds.bbox} /></div>
        <div style={{ textAlign: "center" }}><CapMark value={ds.seg} /></div>
        <div><Badge tone="primary">{ds.access}</Badge></div>
        <div><Badge tone={ds.role.tone}>{ds.role.label}</Badge></div>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--ink-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? "rotate(180deg)" : "rotate(0)", transition: "transform var(--transition)" }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && (
        <div style={{ padding: "var(--sp-5) 1.15rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--sp-6)" }}>
          <div>
            <window.Eyebrow color="var(--ink-3)">Statistics</window.Eyebrow>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--sp-2)", margin: "var(--sp-3) 0 var(--sp-5)" }}>
              {ds.stats.map((s, i) => <StatCard key={i} {...s} />)}
            </div>
            <window.Eyebrow color="var(--ink-3)">Annotations</window.Eyebrow>
            <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginTop: "var(--sp-3)" }}>
              {ds.annos.map((a, i) => <Tag key={i}>{a}</Tag>)}
            </div>
          </div>
          <div>
            <window.Eyebrow color="var(--ink-3)">Key nuances</window.Eyebrow>
            <ul style={{ margin: "var(--sp-3) 0 0", paddingLeft: "1.1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {ds.nuances.map((n, i) => (
                <li key={i} style={{ color: "var(--ink-2)", fontSize: "var(--text-sm)", lineHeight: "var(--lh-snug)" }}>{n}</li>
              ))}
            </ul>
            <p style={{ marginTop: "var(--sp-4)", fontSize: "var(--text-xs)", color: "var(--ink-3)", fontStyle: "italic" }}>{ds.note}</p>
            <a href={ds.url} target="_blank" rel="noreferrer" style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem", marginTop: "var(--sp-3)", fontFamily: "var(--font-mono)", fontSize: "0.74rem", color: "var(--primary)", textDecoration: "none" }}>
              Open dataset <window.Icon name="arrow-up-right" size="0.85rem" />
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

function DataPage() {
  const [openId, setOpenId] = React.useState(window.DATASETS[0].id);
  return (
    <window.Page>
      <window.PageHeader
        eyebrow="Home / Data"
        title="Dataset research"
        lead="Public datasets relevant to TB detection from chest X-rays. Capability marks show bounding-box and segmentation-mask coverage. Click any row for statistics, annotation details and the nuances that bite."
      />

      <window.SectionTitle kicker="Survey" right={
        <div style={{ display: "flex", gap: "1.1rem", fontSize: "var(--text-xs)", color: "var(--ink-3)", alignItems: "center" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}><CapMark value="yes" /> yes</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}><CapMark value="partial" /> partial</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}><CapMark value="no" /> no</span>
        </div>
      }>Public TB &amp; CXR datasets</window.SectionTitle>

      {/* column header */}
      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 0.95fr 0.5fr 0.5fr 0.7fr 0.8fr auto", gap: "1rem", padding: "0 1.15rem 0.7rem", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--ink-3)", fontWeight: "var(--w-semibold)" }}>
        <span>Dataset</span><span>TB images</span><span style={{ textAlign: "center" }}>BBox</span><span style={{ textAlign: "center" }}>Seg</span><span>Access</span><span>Role</span><span></span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
        {window.DATASETS.map((ds) => (
          <DatasetRow key={ds.id} ds={ds} open={openId === ds.id} onToggle={() => setOpenId(openId === ds.id ? null : ds.id)} />
        ))}
      </div>
    </window.Page>
  );
}

window.DataPage = DataPage;
