const { Button, Badge } = window.TBDetectionResearchDesignSystem_ccf532;

// React-native icon set (Lucide path data rendered inside React's own tree —
// no post-render DOM mutation, so reconciliation stays intact).
const ICON_PATHS = {
  home: <><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></>,
  "flask-conical": <><path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2" /><path d="M8.5 2h7" /><path d="M7 16h10" /></>,
  database: <><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5V19A9 3 0 0 0 21 19V5" /><path d="M3 12A9 3 0 0 0 21 12" /></>,
  "book-open": <><path d="M12 7v14" /><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z" /></>,
  "arrow-right": <><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></>,
  "arrow-up-right": <><path d="M7 7h10v10" /><path d="M7 17 17 7" /></>,
  "circle-dashed": <><path d="M10.1 2.18a9.93 9.93 0 0 1 3.8 0" /><path d="M17.6 3.71a9.95 9.95 0 0 1 2.69 2.7" /><path d="M21.82 10.1a9.93 9.93 0 0 1 0 3.8" /><path d="M20.29 17.6a9.95 9.95 0 0 1-2.7 2.69" /><path d="M13.9 21.82a9.94 9.94 0 0 1-3.8 0" /><path d="M6.4 20.29a9.95 9.95 0 0 1-2.69-2.7" /><path d="M2.18 13.9a9.93 9.93 0 0 1 0-3.8" /><path d="M3.71 6.4a9.95 9.95 0 0 1 2.7-2.69" /></>,
  check: <path d="M20 6 9 17l-5-5" />,
  minus: <path d="M5 12h14" />,
  x: <><path d="M18 6 6 18" /><path d="m6 6 12 12" /></>,
  "git-branch": <><line x1="6" x2="6" y1="3" y2="15" /><circle cx="18" cy="6" r="3" /><circle cx="6" cy="18" r="3" /><path d="M18 9a9 9 0 0 1-9 9" /></>,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2" /><path d="M12 20v2" /><path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" /><path d="M2 12h2" /><path d="M20 12h2" /><path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" /></>,
  moon: <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />,
};

function Icon({ name, size = "1em", color = "currentColor", strokeWidth = 2, style = {} }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color}
      strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round"
      style={{ flexShrink: 0, display: "inline-block", verticalAlign: "middle", ...style }}>
      {ICON_PATHS[name] || null}
    </svg>
  );
}

const NAV = [
  { id: "overview", label: "Home", icon: "home" },
  { id: "results", label: "Results & ablation", icon: "flask-conical" },
  { id: "data", label: "Data", icon: "database" },
  { id: "references", label: "References", icon: "book-open" },
];

function Wordmark({ onClick }) {
  return (
    <button onClick={onClick} style={{ all: "unset", cursor: "pointer", display: "flex", alignItems: "center", gap: "0.6rem" }}>
      <span style={{
        width: "32px", height: "32px", borderRadius: "var(--r-sm)", background: "var(--primary)",
        color: "var(--on-primary)", display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: "var(--font-mono)", fontWeight: "var(--w-bold)", fontSize: "0.85rem", letterSpacing: "-0.04em",
      }}>TB</span>
      <span style={{ fontFamily: "var(--font-serif)", fontWeight: "var(--w-semibold)", fontSize: "1.12rem", letterSpacing: "var(--ls-tight)", color: "var(--ink)" }}>
        TB Detection
      </span>
    </button>
  );
}

function ThemeToggle({ theme, setTheme }) {
  const dark = theme === "dark";
  return (
    <button
      onClick={() => setTheme(dark ? "light" : "dark")}
      title={dark ? "Switch to light" : "Switch to dark"}
      aria-label="Toggle colour theme"
      style={{
        all: "unset", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
        width: "34px", height: "34px", borderRadius: "var(--r-pill)", border: "1px solid var(--line)",
        color: "var(--ink-2)", background: "transparent", transition: "background var(--transition), color var(--transition)",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "var(--paper-2)"; e.currentTarget.style.color = "var(--primary)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--ink-2)"; }}
    >
      <Icon name={dark ? "sun" : "moon"} size="1.05rem" />
    </button>
  );
}

function TopBar({ page, setPage, theme, setTheme }) {
  return (
    <header style={{
      position: "sticky", top: 0, zIndex: 50,
      background: "color-mix(in srgb, var(--paper) 82%, transparent)", backdropFilter: "blur(10px)", WebkitBackdropFilter: "blur(10px)",
      borderBottom: "1px solid var(--line)",
    }}>
      <div style={{ maxWidth: "var(--content-max)", margin: "0 auto", padding: "0 var(--gutter)", height: "64px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1.5rem" }}>
        <Wordmark onClick={() => setPage("overview")} />
        <nav style={{ display: "flex", alignItems: "center", gap: "0.15rem" }}>
          {NAV.map((n) => {
            const on = n.id === page;
            return (
              <button key={n.id} className={"nav-link" + (on ? " active" : "")} onClick={() => setPage(n.id)}>
                <Icon name={n.icon} size="1rem" style={{ opacity: 0.85 }} />
                <span style={{ whiteSpace: "nowrap" }}>{n.label}</span>
              </button>
            );
          })}
          <a href="https://github.com/IshuIsAwake/tb_detection_research" target="_blank" rel="noreferrer"
            style={{ marginLeft: "0.5rem", marginRight: "0.5rem", display: "flex", alignItems: "center", gap: "0.35rem", color: "var(--ink-3)", fontSize: "var(--text-sm)", textDecoration: "none" }}
            title="Repository">
            <Icon name="git-branch" size="1.05rem" />
          </a>
          <ThemeToggle theme={theme} setTheme={setTheme} />
        </nav>
      </div>
    </header>
  );
}

function Eyebrow({ children, color = "var(--primary)" }) {
  return (
    <div style={{ fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", fontWeight: "var(--w-semibold)", color }}>
      {children}
    </div>
  );
}

function PageHeader({ eyebrow, title, lead }) {
  return (
    <header style={{ padding: "var(--sp-8) 0 var(--sp-6)", borderBottom: "1px solid var(--line)", marginBottom: "var(--sp-7)" }}>
      <Eyebrow>{eyebrow}</Eyebrow>
      <h1 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-display)", fontWeight: "var(--w-semibold)", letterSpacing: "var(--ls-tight)", lineHeight: "var(--lh-tight)", margin: "0.6rem 0 0" }}>
        {title}
      </h1>
      {lead && (
        <p style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-lead)", color: "var(--ink-2)", lineHeight: "var(--lh-body)", maxWidth: "var(--prose-max)", margin: "1rem 0 0" }}>
          {lead}
        </p>
      )}
    </header>
  );
}

function SectionTitle({ kicker, children, right }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: "1rem", marginBottom: "var(--sp-5)" }}>
      <div>
        {kicker && <Eyebrow color="var(--ink-3)">{kicker}</Eyebrow>}
        <h2 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h2)", fontWeight: "var(--w-semibold)", margin: kicker ? "0.4rem 0 0" : 0, lineHeight: "var(--lh-snug)" }}>
          {children}
        </h2>
      </div>
      {right}
    </div>
  );
}

function Page({ children }) {
  return <main style={{ maxWidth: "var(--content-max)", margin: "0 auto", padding: "0 var(--gutter) var(--sp-9)" }}>{children}</main>;
}

function App() {
  const [page, setPage] = React.useState("overview");
  const [theme, setTheme] = React.useState(() => {
    try { return localStorage.getItem("tb-theme") || "light"; } catch (e) { return "light"; }
  });

  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("tb-theme", theme); } catch (e) {}
  }, [theme]);

  React.useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [page]);

  const Pages = {
    overview: window.OverviewPage,
    results: window.ResultsPage,
    data: window.DataPage,
    references: window.ReferencesPage,
  };
  const Current = Pages[page];

  return (
    <div style={{ minHeight: "100vh", background: "var(--paper)" }}>
      <TopBar page={page} setPage={setPage} theme={theme} setTheme={setTheme} />
      <Current setPage={setPage} />
      <footer style={{ borderTop: "1px solid var(--line)", marginTop: "var(--sp-8)" }}>
        <div style={{ maxWidth: "var(--content-max)", margin: "0 auto", padding: "var(--sp-6) var(--gutter)", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem", color: "var(--ink-3)", fontSize: "var(--text-sm)" }}>
          <span>TB detection on chest X-rays — a working research notebook.</span>
          <a href="https://github.com/IshuIsAwake/tb_detection_research" target="_blank" rel="noreferrer" style={{ color: "var(--primary)", textDecoration: "none", fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>
            github.com/IshuIsAwake/tb_detection_research
          </a>
        </div>
      </footer>
    </div>
  );
}

Object.assign(window, { App, Page, PageHeader, SectionTitle, Eyebrow, Wordmark, Icon });
