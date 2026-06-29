const { StatCard, Callout, Badge, Tag, Button, Tabs } = window.TBDetectionResearchDesignSystem_ccf532;

const COLS = "1.7fr 0.7fr 0.9fr 0.72fr 0.95fr 1.05fr 0.85fr 0.8fr 34px";

// tone helpers for the colour-coded metric boxes
const t = {
  map: (v) => (parseFloat(v) >= 0.5 ? "good" : parseFloat(v) >= 0.42 ? "warn" : "bad"),
  prec: (v) => (parseFloat(v) >= 0.5 ? "good" : parseFloat(v) >= 0.35 ? "warn" : "bad"),
  rec: (v) => (parseFloat(v) >= 0.5 ? "good" : parseFloat(v) >= 0.45 ? "warn" : "bad"),
  detect: (v) => (parseFloat(v) >= 90 ? "good" : parseFloat(v) >= 80 ? "warn" : "bad"),
};

function MiniHead({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", margin: "0 0 var(--sp-3)" }}>
      <span style={{ fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", fontWeight: "var(--w-semibold)", color: "var(--ink-3)", whiteSpace: "nowrap" }}>{children}</span>
      <span style={{ flex: 1, height: "1px", background: "var(--line)" }} />
    </div>
  );
}

function MetricGrid({ children, min = "140px" }) {
  return <div style={{ display: "grid", gridTemplateColumns: `repeat(auto-fill, minmax(${min}, 1fr))`, gap: "var(--sp-3)", marginBottom: "var(--sp-5)" }}>{children}</div>;
}

function RunMetrics({ run }) {
  const m = window.buildRunMetrics(run);
  const [thr, setThr] = React.useState("0.25");
  const sc = m.screening[thr];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "var(--sp-6)" }}>
      <div>
        <MiniHead>Detection</MiniHead>
        <MetricGrid>
          <StatCard label="mAP50 (overall)" value={m.overall} tone={t.map(m.overall)} />
          <StatCard label="mAP50-95" value={m.map5095} />
          <StatCard label="Precision" value={m.precision} tone={t.prec(m.precision)} />
          <StatCard label="Recall" value={m.recall} tone={t.rec(m.recall)} />
        </MetricGrid>
        <MiniHead>Active TB</MiniHead>
        <MetricGrid><StatCard label="mAP50" value={m.active} tone="good" /></MetricGrid>
        <MiniHead>Obsolete TB</MiniHead>
        <MetricGrid><StatCard label="mAP50" value={m.obsolete} tone="bad" note="class imbalance — barely learnable" /></MetricGrid>
      </div>
      <div>
        <MiniHead>Localization (matched pairs)</MiniHead>
        <MetricGrid><StatCard label="IoU" value={m.iou} tone="good" note="boxes are well-placed when found" /></MetricGrid>
        <MiniHead>Screening</MiniHead>
        <div style={{ marginBottom: "var(--sp-3)" }}>
          <Tabs items={[{ id: "0.10", label: "@0.10" }, { id: "0.25", label: "@0.25" }, { id: "0.50", label: "@0.50" }]} value={thr} onChange={setThr} />
        </div>
        <MetricGrid>
          <StatCard label="TB detect rate" value={sc.detect} tone={t.detect(sc.detect)} />
          <StatCard label="TB flagged" value={sc.flagged} />
          <StatCard label="Healthy FA" value={sc.healthyFA} tone={window.faTone(sc.healthyFA, true)} />
          <StatCard label="Sick FA" value={sc.sickFA} tone={window.faTone(sc.sickFA, false)} />
        </MetricGrid>
        <MiniHead>Recall by lesion size</MiniHead>
        <MetricGrid>
          <StatCard label={`Small (n=${m.lesion.small.n})`} value={m.lesion.small.v} tone="bad" />
          <StatCard label={`Medium (n=${m.lesion.medium.n})`} value={m.lesion.medium.v} tone="warn" />
          <StatCard label={`Large (n=${m.lesion.large.n})`} value={m.lesion.large.v} tone="good" />
        </MetricGrid>
      </div>
    </div>
  );
}

function RunSettings({ run }) {
  const rows = window.buildRunSettings(run);
  return (
    <div>
      <MiniHead>Training configuration</MiniHead>
      <MetricGrid min="150px">
        {rows.map((s, i) => <StatCard key={i} label={s.label} value={s.value} />)}
      </MetricGrid>
    </div>
  );
}

function FAcell({ run }) {
  return (
    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem", fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>
      <span style={{ color: `var(--${window.faTone(run.healthyFA, true)})`, fontWeight: "var(--w-semibold)" }}>{run.healthyFA}</span>
      <span style={{ color: "var(--ink-4)" }}> / </span>
      <span style={{ color: `var(--${window.faTone(run.sickFA, false)})` }}>{run.sickFA}</span>
    </span>
  );
}

function Num({ v, tone }) {
  const color = tone ? `var(--${tone})` : "var(--ink)";
  return <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", fontVariantNumeric: "tabular-nums", color, fontWeight: tone === "good" ? "var(--w-semibold)" : "var(--w-regular)" }}>{v}</span>;
}

function RunRow({ run }) {
  const [open, setOpen] = React.useState(false);
  const [tab, setTab] = React.useState("description");
  return (
    <div style={{ borderBottom: "1px solid var(--line)" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{ all: "unset", boxSizing: "border-box", cursor: "pointer", width: "100%", display: "grid", gridTemplateColumns: COLS, alignItems: "center", gap: "0.6rem", padding: "0.95rem 1.1rem", background: open ? "var(--paper-2)" : "transparent", transition: "background var(--transition)" }}
        onMouseEnter={(e) => { if (!open) e.currentTarget.style.background = "var(--surface-2)"; }}
        onMouseLeave={(e) => { if (!open) e.currentTarget.style.background = "transparent"; }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: "0.5rem", minWidth: 0 }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem", fontWeight: "var(--w-semibold)", color: "var(--ink)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{run.name}</span>
          {run.best && <Badge tone="primary">best</Badge>}
        </span>
        <Num v={run.map50} tone={t.map(run.map50)} />
        <Num v={run.precision} />
        <Num v={run.recall} tone={parseFloat(run.recall) >= 0.5 ? "good" : null} />
        <Num v={run.tbCaught} />
        <FAcell run={run} />
        <Num v={run.trainTime} />
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "0.76rem", color: "var(--ink-3)", whiteSpace: "nowrap" }}>{window.RUN_DATE}</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--ink-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ justifySelf: "end", transform: open ? "rotate(180deg)" : "rotate(0)", transition: "transform var(--transition)" }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && (
        <div style={{ padding: "var(--sp-4) 1.1rem var(--sp-6)", background: "var(--surface)", borderTop: "1px solid var(--line)" }}>
          <div style={{ marginBottom: "var(--sp-5)" }}>
            <Tabs items={[{ id: "description", label: "Description" }, { id: "metrics", label: "Metrics" }, { id: "settings", label: "Settings" }]} value={tab} onChange={setTab} />
          </div>
          {tab === "description" && (
            <div style={{ background: "var(--paper-2)", border: "1px solid var(--line)", borderRadius: "var(--r)", padding: "var(--sp-5)", maxWidth: "var(--prose-max)" }}>
              <p style={{ margin: 0, fontFamily: "var(--font-sans)", fontSize: "var(--text-body)", color: "var(--ink-2)", lineHeight: "var(--lh-body)", textWrap: "pretty" }}>{run.desc}</p>
            </div>
          )}
          {tab === "metrics" && <RunMetrics run={run} />}
          {tab === "settings" && <RunSettings run={run} />}
        </div>
      )}
    </div>
  );
}

function YoloExperiments() {
  const [open, setOpen] = React.useState(true);
  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: "var(--r-lg)", background: "var(--surface)", boxShadow: "var(--shadow-sm)", overflow: "hidden" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{ all: "unset", boxSizing: "border-box", cursor: "pointer", width: "100%", display: "flex", alignItems: "center", gap: "0.9rem", padding: "1rem 1.2rem", background: "var(--paper-2)", borderBottom: open ? "1px solid var(--line)" : "1px solid transparent" }}
      >
        <span style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h3)", fontWeight: "var(--w-semibold)", color: "var(--ink)" }}>YOLO Experiments</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--ink-3)" }}>{window.RUNS.length} runs · newest first</span>
        <span style={{ flex: 1 }} />
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--ink-3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? "rotate(180deg)" : "rotate(0)", transition: "transform var(--transition)" }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && (
        <div>
          {/* column header */}
          <div style={{ display: "grid", gridTemplateColumns: COLS, gap: "0.6rem", padding: "0.7rem 1.1rem", borderBottom: "1px solid var(--line-2)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--ink-3)", fontWeight: "var(--w-semibold)" }}>
            <span>Experiment</span><span>mAP50</span><span>Precision</span><span>Recall</span><span>TB caught @.25</span><span>FA @.25 (H/S)</span><span>Train time</span><span>Date</span><span></span>
          </div>
          {window.RUNS.map((run) => <RunRow key={run.name} run={run} />)}
        </div>
      )}
    </div>
  );
}

function LatestRun() {
  const latest = window.RUNS[0];
  const m = window.buildRunMetrics(latest);
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: "var(--r-lg)", padding: "var(--sp-6)", boxShadow: "var(--shadow-sm)" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.75rem", flexWrap: "wrap", marginBottom: "0.5rem" }}>
        <h3 style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-h3)", fontWeight: "var(--w-semibold)", margin: 0, color: "var(--primary)" }}>{latest.name}</h3>
        <Badge tone="primary" variant="solid">Latest run</Badge>
        {latest.best && <Badge tone="good">augmentation finalist</Badge>}
      </div>
      <p style={{ color: "var(--ink-2)", fontSize: "var(--text-body)", lineHeight: "var(--lh-body)", margin: "0 0 1.25rem", maxWidth: "var(--prose-max)", textWrap: "pretty" }}>
        Final detector config: <Tag tone="primary">mosaic @ 512, batch 16</Tag>. mosaic is the augmentation
        finalist — its active-mAP50 bands across seeds don't overlap with geometry-only.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: "var(--sp-3)" }}>
        <StatCard label="Active mAP50" value={m.active} tone="good" />
        <StatCard label="Recall" value={latest.recall} tone="good" />
        <StatCard label="TB caught @.25" value={latest.tbCaught} />
        <StatCard label="Healthy FA @.25" value={latest.healthyFA} tone={window.faTone(latest.healthyFA, true)} />
        <StatCard label="Matched IoU" value={m.iou} tone="good" />
      </div>
    </div>
  );
}

function ResultsPage() {
  return (
    <window.Page>
      <header style={{ padding: "var(--sp-8) 0 var(--sp-6)", borderBottom: "1px solid var(--line)", marginBottom: "var(--sp-7)" }}>
        <h1 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-display)", fontWeight: "var(--w-semibold)", letterSpacing: "var(--ls-tight)", lineHeight: "var(--lh-tight)", margin: 0 }}>
          Results &amp; ablation
        </h1>
      </header>

      {/* Note box */}
      <Callout kind="note" title="Note" style={{ marginBottom: "var(--sp-7)" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          <div><strong style={{ color: "var(--ink)" }}>a.</strong> We optimize mAP and lesion-level recall while balancing localization.</div>
          <div><strong style={{ color: "var(--ink)" }}>b.</strong> On non-CV runs we use a frozen split. On CV runs the test set is always disjoint. <span style={{ color: "var(--ink-3)" }}>(CV = cross-validation)</span></div>
        </div>
      </Callout>

      {/* Latest run */}
      <section style={{ marginBottom: "var(--sp-8)" }}>
        <window.SectionTitle>Latest run</window.SectionTitle>
        <LatestRun />
      </section>

      {/* Experiments */}
      <section>
        <window.SectionTitle>Experiments</window.SectionTitle>
        <YoloExperiments />
      </section>

      {/* Next */}
      <div style={{ marginTop: "var(--sp-7)" }}>
        <window.SectionTitle kicker="Planned">Next</window.SectionTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
          {window.NEXT_UP.map((x) => (
            <div key={x.id} style={{ display: "flex", gap: "0.85rem", alignItems: "center", padding: "0.9rem 1.1rem", background: "var(--paper-2)", border: "1px solid var(--line)", borderRadius: "var(--r)" }}>
              <Tag tone="primary">{x.id}</Tag>
              <span style={{ color: "var(--ink-2)", fontSize: "var(--text-sm)" }} dangerouslySetInnerHTML={{ __html: x.text }} />
            </div>
          ))}
        </div>
      </div>
    </window.Page>
  );
}

window.ResultsPage = ResultsPage;
