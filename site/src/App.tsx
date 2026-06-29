import React from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { TopBar } from "@/components/Shell";
import { OverviewPage }    from "@/pages/OverviewPage";
import { ResultsPage }     from "@/pages/ResultsPage";
import { DataPage }        from "@/pages/DataPage";
import { ReferencesPage }  from "@/pages/ReferencesPage";

function ScrollToTop() {
  const { pathname } = useLocation();
  React.useEffect(() => { window.scrollTo({ top: 0 }); }, [pathname]);
  return null;
}

export function App() {
  return (
    <div style={{ minHeight: "100vh", background: "var(--paper)" }}>
      <ScrollToTop />
      <TopBar />
      <Routes>
        <Route path="/"           element={<OverviewPage />} />
        <Route path="/results"    element={<ResultsPage />} />
        <Route path="/data"       element={<DataPage />} />
        <Route path="/references" element={<ReferencesPage />} />
      </Routes>
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
