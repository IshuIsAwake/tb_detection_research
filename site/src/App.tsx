import React from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { Sidebar, SIDEBAR_W, MOBILE_BP } from "@/components/Sidebar";
import { Icon } from "@/components/Icon";
import { OverviewPage } from "@/pages/OverviewPage";
import { AboutTBPage } from "@/pages/AboutTBPage";
import { HowDetectedPage } from "@/pages/HowDetectedPage";
import { ActiveTBDetectionPage } from "@/pages/ActiveTBDetectionPage";
import { HistoryCADPage } from "@/pages/HistoryCADPage";
import { ArchitecturesPage } from "@/pages/ArchitecturesPage";
import { ResultsPage } from "@/pages/ResultsPage";
import { DataPage } from "@/pages/DataPage";
import { ReferencesPage } from "@/pages/ReferencesPage";

function ScrollToTop() {
  const { pathname } = useLocation();
  React.useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [pathname]);
  return null;
}

/** Track the viewport against the sidebar's breakpoint. */
function useIsMobile() {
  const [mobile, setMobile] = React.useState(
    () => typeof window !== "undefined" && window.innerWidth < MOBILE_BP,
  );
  React.useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${MOBILE_BP - 1}px)`);
    const on = () => setMobile(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return mobile;
}

function Footer() {
  return (
    <footer style={{ borderTop: "1px solid var(--line)", marginTop: "var(--sp-8)" }}>
      <div
        style={{
          maxWidth: "var(--content-max)",
          margin: "0 auto",
          padding: "var(--sp-6) var(--gutter)",
          display: "flex",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "1rem",
          color: "var(--ink-3)",
          fontSize: "var(--text-sm)",
        }}
      >
        <span>TB detection on chest X-rays — a working research notebook.</span>
        <a
          href="https://github.com/IshuIsAwake/tb_detection_research"
          target="_blank"
          rel="noreferrer"
          style={{
            color: "var(--primary)",
            textDecoration: "none",
            fontFamily: "var(--font-mono)",
            fontSize: "0.78rem",
          }}
        >
          github.com/IshuIsAwake/tb_detection_research
        </a>
      </div>
    </footer>
  );
}

/** The button that reopens a collapsed sidebar (and the mobile hamburger). */
function OpenButton({ onClick, mobile }: { onClick: () => void; mobile: boolean }) {
  return (
    <button
      onClick={onClick}
      aria-label="Open contents"
      title="Open contents"
      style={{
        position: "fixed",
        top: mobile ? "0.75rem" : "0.9rem",
        left: mobile ? "0.75rem" : "0.9rem",
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        gap: "0.45rem",
        padding: "0.45rem 0.7rem",
        borderRadius: "var(--r)",
        border: "1px solid var(--line)",
        background: "color-mix(in srgb, var(--surface) 88%, transparent)",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        color: "var(--ink-2)",
        cursor: "pointer",
        fontFamily: "var(--font-sans)",
        fontSize: "var(--text-sm)",
        fontWeight: "var(--w-medium)",
        boxShadow: "var(--shadow-sm)",
      }}
    >
      <Icon name={mobile ? "menu" : "panel-left-open"} size="1.05rem" />
      {!mobile && <span>Contents</span>}
    </button>
  );
}

export function App() {
  const mobile = useIsMobile();
  const [open, setOpen] = React.useState<boolean>(() => {
    try {
      const v = localStorage.getItem("tb-sidebar");
      if (v !== null) return v === "1";
    } catch {
      /* ignore */
    }
    return true;
  });

  // Persist only the desktop preference; on mobile the drawer always starts shut.
  React.useEffect(() => {
    if (mobile) return;
    try {
      localStorage.setItem("tb-sidebar", open ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [open, mobile]);

  React.useEffect(() => {
    if (mobile) setOpen(false);
  }, [mobile]);

  // Escape closes the mobile drawer.
  React.useEffect(() => {
    if (!mobile || !open) return;
    const on = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [mobile, open]);

  return (
    <div style={{ minHeight: "100vh", background: "var(--paper)" }}>
      <ScrollToTop />
      <Sidebar open={open} setOpen={setOpen} mobile={mobile} />
      {!open && <OpenButton onClick={() => setOpen(true)} mobile={mobile} />}

      <div
        style={{
          marginLeft: !mobile && open ? SIDEBAR_W : 0,
          transition: "margin-left var(--dur-slow) var(--ease)",
          minWidth: 0,
        }}
      >
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/about-tb" element={<AboutTBPage />} />
          <Route path="/how-tb-detected" element={<HowDetectedPage />} />
          <Route path="/active-tb-detection" element={<ActiveTBDetectionPage />} />
          <Route path="/history-cad" element={<HistoryCADPage />} />
          <Route path="/architectures" element={<ArchitecturesPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/references" element={<ReferencesPage />} />
        </Routes>
        <Footer />
      </div>
    </div>
  );
}
