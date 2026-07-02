import React from "react";
import {
  READS, ReadingNav, NextSection, SourceList, makeCite,
  InfoCard, CardGrid, Bullets, ReadSection, Prose, type Source,
} from "@/components/Reading";

const SOURCES: Source[] = [
  { label: "WHO — Rapid diagnostics for TB detection (Module 3)", url: "https://www.who.int/publications/i/item/9789240089488" },
  { label: "WHO — Systematic screening & CAD (Module 2)", url: "https://www.who.int/publications/i/item/9789240022676" },
];

const Cite = makeCite(SOURCES);

const COMPARE: { method: React.ReactNode; cost: string; speed: string; accuracy: string; access: string; x: string }[] = [
  { method: <>Rapid molecular assays (GeneXpert)<Cite n={1} /></>, cost: "Moderate / high", speed: "< 2 hours", accuracy: "High — detects DNA & resistance", access: "Moderate", x: "Flags drug resistance immediately; 2026 tongue-swab protocols" },
  { method: <>Sputum smear microscopy<Cite n={1} /></>, cost: "Very low ($2–3)", speed: "Fast (hours)", accuracy: "Low sensitivity", access: "Very high", x: "Frontline tool; needs no advanced infrastructure" },
  { method: <>Mycobacterial culture<Cite n={1} /></>, cost: "High", speed: "Very slow (2–6 weeks)", accuracy: "Gold standard (~100%)", access: "Low", x: "Complete drug-susceptibility profiling" },
  { method: <>Chest X-ray with AI/CAD<Cite n={2} /></>, cost: "Moderate", speed: "Very fast (minutes)", accuracy: "High sensitivity, low specificity", access: "Moderate", x: "AI screening at scale; instantly flags high-probability cases" },
];

const HEAD = ["Method", "Cost", "Speed", "Accuracy", "Accessibility", "X-factor"];

function Th({ children }: { children: React.ReactNode }) {
  return <th style={{ textAlign: "left", whiteSpace: "nowrap", padding: "0.7rem 0.9rem", fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--ink-3)", fontWeight: "var(--w-semibold)", borderBottom: "1px solid var(--line)" }}>{children}</th>;
}
function Td({ children, strong }: { children: React.ReactNode; strong?: boolean }) {
  return <td style={{ verticalAlign: "top", padding: "0.75rem 0.9rem", fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: strong ? "var(--ink)" : "var(--ink-2)", fontWeight: strong ? "var(--w-semibold)" : "var(--w-regular)", lineHeight: "var(--lh-snug)", borderTop: "1px solid var(--line)" }}>{children}</td>;
}

export function ActiveTBDetectionPage() {
  const read = READS[2];
  return (
    <main style={{ maxWidth: "var(--content-max)", margin: "0 auto", padding: "var(--sp-7) var(--gutter) var(--sp-9)" }}>
      <ReadingNav current={read.n} />

      <header style={{ marginBottom: "var(--sp-6)" }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--primary)", fontWeight: "var(--w-semibold)" }}>Read {read.n} of {READS.length}</div>
        <h1 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-display)", fontWeight: "var(--w-semibold)", letterSpacing: "var(--ls-tight)", lineHeight: "var(--lh-tight)", margin: "0.5rem 0 0" }}>How is active TB detected?</h1>
        <Prose style={{ marginTop: "var(--sp-4)", marginBottom: 0 }}>
          A closer look at the four primary methods used to diagnose and screen for active tuberculosis — how each
          one works, and what it trades off.
        </Prose>
      </header>

      <ReadSection n={1} title="At a glance">
        <div style={{ border: "1px solid var(--line)", borderRadius: "var(--r)", overflowX: "auto", background: "var(--surface)" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
            <thead><tr style={{ background: "var(--paper-2)" }}>{HEAD.map((h) => <Th key={h}>{h}</Th>)}</tr></thead>
            <tbody>
              {COMPARE.map((r, i) => (
                <tr key={i}>
                  <Td strong>{r.method}</Td>
                  <Td>{r.cost}</Td>
                  <Td>{r.speed}</Td>
                  <Td>{r.accuracy}</Td>
                  <Td>{r.access}</Td>
                  <Td>{r.x}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ReadSection>

      <ReadSection n={2} title="The four methods">
        <CardGrid cols={2}>
          <InfoCard title="1 · Rapid molecular assays (GeneXpert)">
            <Bullets items={[
              <><strong>Time to result:</strong> under 2 hours.</>,
              <><strong>How it works:</strong> automated PCR amplifies and detects <em>M. tuberculosis</em> DNA directly from a patient sample.</>,
              <><strong>X-factor:</strong> it also detects the mutations that cause Rifampicin resistance, so drug-resistant TB is known on day one. WHO guidance now supports simple tongue swabs instead of deep-lung sputum — a breakthrough for children and severely ill adults.<Cite n={1} /></>,
            ]} />
          </InfoCard>
          <InfoCard title="2 · Sputum smear microscopy">
            <Bullets items={[
              <><strong>Cost:</strong> extremely low, often under $2–3 per test.</>,
              <><strong>How it works:</strong> a stained sputum smear is examined under a standard light microscope for the physical bacteria.</>,
              <><strong>X-factor:</strong> needs no reliable electricity, proprietary cartridges, or advanced lab — the frontline tool in rural, low-resource settings.<Cite n={1} /></>,
              <><strong>Downside:</strong> poor sensitivity; it needs a heavy bacterial load and frequently misses TB in children and people co-infected with HIV.</>,
            ]} />
          </InfoCard>
          <InfoCard title="3 · Mycobacterial culture">
            <Bullets items={[
              <><strong>Accuracy:</strong> the definitive gold standard — sensitivity near 100%, specificity absolute.</>,
              <><strong>How it works:</strong> sputum is grown in a controlled nutrient medium; if bacteria are present, they multiply and prove active disease.</>,
              <><strong>X-factor:</strong> the grown bacteria can be exposed to every TB drug for a complete drug-susceptibility profile — the only way to be certain which antibiotics will cure a complex, resistant case.<Cite n={1} /></>,
              <><strong>Downside:</strong> TB grows glacially — a culture takes 2–6 weeks, so doctors cannot wait for it before treating a strong suspicion.</>,
            ]} />
          </InfoCard>
          <InfoCard title="4 · Chest X-ray (with AI/CAD)">
            <Bullets items={[
              <><strong>Accuracy:</strong> very high sensitivity but low specificity — TB damage looks like pneumonia, fungal infection, or cancer, so it is a triage and screening tool, not a definitive diagnosis.</>,
              <><strong>How it works:</strong> images the lungs to show damage, cavities, fluid, or scarring from infection.</>,
              <><strong>X-factor:</strong> reading X-rays used to need a scarce radiologist. The WHO now formally recommends CAD software — AI that scores TB probability in seconds, letting mobile clinics screen hundreds a day and route only flagged cases to a GeneXpert test.<Cite n={2} /></>,
            ]} />
          </InfoCard>
        </CardGrid>
      </ReadSection>

      <SourceList sources={SOURCES} />
      <NextSection current={read.n} />
    </main>
  );
}
