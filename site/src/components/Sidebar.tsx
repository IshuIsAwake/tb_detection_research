import React from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { Icon } from "@/components/Icon";
import { useTheme } from "@/lib/useTheme";
import { NAV_GROUPS } from "@/data/nav";

const SIDEBAR_W = 274;
const RAIL_W = 0;
const MOBILE_BP = 1080;

// ─── Section discovery ────────────────────────────────────────────────────────
// The outline under the active page is read from the DOM rather than duplicated
// in a data file: any element with `data-section="Title"` becomes an entry. The
// page's prose is therefore the single source of truth, and the outline cannot
// drift from it.

interface DiscoveredSection {
  id: string;
  label: string;
  n?: string;
}

function useSections(pathname: string) {
  const [sections, setSections] = React.useState<DiscoveredSection[]>([]);
  const [activeId, setActiveId] = React.useState<string>("");

  React.useEffect(() => {
    let raf = 0;
    const scan = () => {
      const nodes = Array.from(document.querySelectorAll<HTMLElement>("[data-section]"));
      setSections(
        nodes.map((el) => ({
          id: el.id,
          label: el.dataset.section || "",
          n: el.dataset.sectionN,
        })),
      );
      setActiveId(nodes[0]?.id ?? "");
    };
    // Wait one frame so the freshly-routed page has mounted.
    raf = requestAnimationFrame(scan);
    return () => cancelAnimationFrame(raf);
  }, [pathname]);

  React.useEffect(() => {
    if (!sections.length) return;
    const els = sections
      .map((s) => document.getElementById(s.id))
      .filter((e): e is HTMLElement => !!e);
    if (!els.length) return;

    // Track which sections are on screen; the topmost visible one wins. Falling
    // back to "last section scrolled past" keeps a heading highlighted even when
    // a long section fills the viewport with no heading in view.
    const visible = new Set<string>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) visible.add(e.target.id);
          else visible.delete(e.target.id);
        }
        const first = els.find((el) => visible.has(el.id));
        if (first) {
          setActiveId(first.id);
          return;
        }
        const passed = els.filter((el) => el.getBoundingClientRect().top < 120);
        if (passed.length) setActiveId(passed[passed.length - 1].id);
      },
      { rootMargin: "-80px 0px -55% 0px", threshold: 0 },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [sections]);

  return { sections, activeId };
}

// NOTE: the app runs on HashRouter, so `href="#some-id"` would be parsed as a
// ROUTE, not an anchor. In-page navigation must be programmatic.
function scrollToSection(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const top = el.getBoundingClientRect().top + window.scrollY - 72;
  window.scrollTo({ top, behavior: reduce ? "auto" : "smooth" });
}

// ─── Wordmark ─────────────────────────────────────────────────────────────────

function Wordmark({ onNavigate }: { onNavigate?: () => void }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => {
        navigate("/");
        onNavigate?.();
      }}
      style={{
        all: "unset",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        gap: "0.6rem",
        minWidth: 0,
      }}
    >
      <span
        style={{
          flexShrink: 0,
          width: "30px",
          height: "30px",
          borderRadius: "var(--r-sm)",
          background: "var(--primary)",
          color: "var(--on-primary)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "var(--font-mono)",
          fontWeight: "var(--w-bold)",
          fontSize: "0.8rem",
          letterSpacing: "-0.04em",
        }}
      >
        TB
      </span>
      <span style={{ minWidth: 0 }}>
        <span
          style={{
            display: "block",
            fontFamily: "var(--font-serif)",
            fontWeight: "var(--w-semibold)",
            fontSize: "1.02rem",
            letterSpacing: "var(--ls-tight)",
            color: "var(--ink)",
            lineHeight: 1.2,
            whiteSpace: "nowrap",
          }}
        >
          TB Detection
        </span>
        <span
          style={{
            display: "block",
            fontFamily: "var(--font-mono)",
            fontSize: "0.62rem",
            textTransform: "uppercase",
            letterSpacing: "var(--ls-label)",
            color: "var(--ink-3)",
            whiteSpace: "nowrap",
          }}
        >
          Living research paper
        </span>
      </span>
    </button>
  );
}

// ─── Outline rows ─────────────────────────────────────────────────────────────

function SectionRow({
  s,
  active,
  onClick,
}: {
  s: DiscoveredSection;
  active: boolean;
  onClick: () => void;
}) {
  const [hover, setHover] = React.useState(false);
  return (
    <li>
      <button
        onClick={onClick}
        aria-current={active ? "location" : undefined}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          all: "unset",
          boxSizing: "border-box",
          cursor: "pointer",
          display: "flex",
          gap: "0.5rem",
          width: "100%",
          padding: "0.3rem 0.6rem 0.3rem 0",
          // The rail line: an active marker that sits in the indent channel.
          borderLeft: `2px solid ${active ? "var(--primary)" : "transparent"}`,
          paddingLeft: "0.85rem",
          marginLeft: "0.55rem",
          color: active ? "var(--primary)" : hover ? "var(--ink-2)" : "var(--ink-3)",
          fontFamily: "var(--font-sans)",
          fontSize: "0.78rem",
          fontWeight: active ? "var(--w-semibold)" : "var(--w-regular)",
          lineHeight: 1.4,
          transition: "color var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease)",
        }}
      >
        {s.n && (
          <span
            style={{
              flexShrink: 0,
              fontFamily: "var(--font-mono)",
              fontSize: "0.68rem",
              opacity: active ? 0.9 : 0.6,
              paddingTop: "0.08rem",
            }}
          >
            {s.n}
          </span>
        )}
        <span style={{ minWidth: 0 }}>{s.label}</span>
      </button>
    </li>
  );
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

export function Sidebar({
  open,
  setOpen,
  mobile,
}: {
  open: boolean;
  setOpen: (v: boolean) => void;
  mobile: boolean;
}) {
  const { pathname } = useLocation();
  const { theme, toggle } = useTheme();
  const { sections, activeId } = useSections(pathname);

  const body = (
    <>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
          padding: "0.9rem 0.9rem 0.9rem 1rem",
          borderBottom: "1px solid var(--line)",
          minHeight: "62px",
          boxSizing: "border-box",
        }}
      >
        <Wordmark onNavigate={() => mobile && setOpen(false)} />
        <button
          onClick={() => setOpen(false)}
          aria-label="Collapse sidebar"
          title="Collapse sidebar"
          className="icon-btn"
        >
          <Icon name={mobile ? "x" : "panel-left-close"} size="1.05rem" />
        </button>
      </div>

      {/* Outline */}
      <nav aria-label="Contents" style={{ flex: 1, overflowY: "auto", padding: "0.7rem 0.6rem 1rem" }}>
        {NAV_GROUPS.map((g) => (
          <div key={g.id} style={{ marginBottom: "1.1rem" }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.62rem",
                textTransform: "uppercase",
                letterSpacing: "var(--ls-label)",
                color: "var(--ink-4)",
                fontWeight: "var(--w-semibold)",
                padding: "0 0.5rem",
                margin: "0 0 0.4rem",
              }}
            >
              {g.label}
            </div>
            <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
              {g.pages.map((p) => {
                const isActive = pathname === p.path;
                return (
                  <li key={p.path}>
                    <NavLink
                      to={p.path}
                      onClick={() => mobile && setOpen(false)}
                      className={({ isActive: a }) => "outline-link" + (a ? " active" : "")}
                    >
                      {p.n ? (
                        <span className="outline-num">{p.n}</span>
                      ) : (
                        <Icon name={p.icon} size="0.92rem" style={{ opacity: 0.75, flexShrink: 0 }} />
                      )}
                      <span style={{ minWidth: 0 }}>{p.label}</span>
                    </NavLink>

                    {/* Sub-sections, discovered from the live page */}
                    {isActive && sections.length > 0 && (
                      <ul style={{ listStyle: "none", margin: "0.15rem 0 0.5rem", padding: 0 }}>
                        {sections.map((s) => (
                          <SectionRow
                            key={s.id}
                            s={s}
                            active={s.id === activeId}
                            onClick={() => {
                              scrollToSection(s.id);
                              if (mobile) setOpen(false);
                            }}
                          />
                        ))}
                      </ul>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div
        style={{
          borderTop: "1px solid var(--line)",
          padding: "0.6rem 0.7rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.5rem",
        }}
      >
        <a
          href="https://github.com/IshuIsAwake/tb_detection_research"
          target="_blank"
          rel="noreferrer"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
            color: "var(--ink-3)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.68rem",
            textDecoration: "none",
            padding: "0.3rem 0.4rem",
            borderRadius: "var(--r-sm)",
          }}
          title="Source repository"
        >
          <Icon name="git-branch" size="0.95rem" />
          <span>Source</span>
        </a>
        <button
          onClick={toggle}
          className="icon-btn"
          aria-label="Toggle colour theme"
          title={theme === "dark" ? "Switch to light" : "Switch to dark"}
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} size="1rem" />
        </button>
      </div>
    </>
  );

  if (mobile) {
    return (
      <>
        {open && (
          <div
            onClick={() => setOpen(false)}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(20,22,24,0.44)",
              backdropFilter: "blur(2px)",
              zIndex: 60,
            }}
          />
        )}
        <aside
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            bottom: 0,
            width: SIDEBAR_W,
            maxWidth: "86vw",
            display: "flex",
            flexDirection: "column",
            background: "var(--surface)",
            borderRight: "1px solid var(--line)",
            zIndex: 61,
            transform: open ? "translateX(0)" : `translateX(-${SIDEBAR_W + 8}px)`,
            transition: "transform var(--dur-slow) var(--ease)",
            boxShadow: open ? "var(--shadow-lg)" : "none",
          }}
        >
          {body}
        </aside>
      </>
    );
  }

  return (
    <aside
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        bottom: 0,
        width: open ? SIDEBAR_W : RAIL_W,
        display: "flex",
        flexDirection: "column",
        background: "var(--paper-2)",
        borderRight: open ? "1px solid var(--line)" : "none",
        overflow: "hidden",
        zIndex: 40,
        transition: "width var(--dur-slow) var(--ease)",
      }}
    >
      <div style={{ width: SIDEBAR_W, display: "flex", flexDirection: "column", height: "100%" }}>
        {body}
      </div>
    </aside>
  );
}

export { SIDEBAR_W, MOBILE_BP };
