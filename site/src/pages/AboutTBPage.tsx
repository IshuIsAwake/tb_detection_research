import React from "react";
import {
  READS, ReadingNav, NextSection, SourceList, makeCite, Collapsible,
  InfoCard, CardGrid, Bullets, ReadSection, Prose, type Source,
} from "@/components/Reading";

const SOURCES: Source[] = [
  { label: "WHO — Tuberculosis fact sheet (2024 figures)", url: "https://www.who.int/news-room/fact-sheets/detail/tuberculosis" },
  { label: "WHO — Tuberculosis (health topic)", url: "https://www.who.int/health-topics/tuberculosis" },
  { label: "CDC — Combating global TB", url: "https://www.cdc.gov/global-hiv-tb/php/our-approach/combatingglobaltb.html" },
  { label: "Treatment outcomes of MDR-TB (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11995120/" },
  { label: "WHO Global TB Report 2025 — summary (EATG)", url: "https://www.eatg.org/hiv-news/who-global-tb-report-2025-global-gains-in-tb-response-endangered-by-funding-challenges/" },
  { label: "WHO Global Tuberculosis Report 2025 (ReliefWeb)", url: "https://reliefweb.int/report/world/global-tuberculosis-report-2025" },
];

const Cite = makeCite(SOURCES);

function TableRow({ head, children, first }: { head: string; children: React.ReactNode; first?: boolean }) {
  return (
    <tr>
      <th scope="row" style={{ textAlign: "left", verticalAlign: "top", width: "34%", padding: "0.85rem 1.1rem", borderTop: first ? "none" : "1px solid var(--line)", fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", fontWeight: "var(--w-semibold)", color: "var(--ink)" }}>{head}</th>
      <td style={{ verticalAlign: "top", padding: "0.85rem 1.1rem", borderTop: first ? "none" : "1px solid var(--line)", fontFamily: "var(--font-sans)", fontSize: "var(--text-sm)", color: "var(--ink-2)", lineHeight: "var(--lh-snug)" }}>{children}</td>
    </tr>
  );
}

export function AboutTBPage() {
  const read = READS[0];
  return (
    <main style={{ maxWidth: "var(--content-max)", margin: "0 auto", padding: "var(--sp-7) var(--gutter) var(--sp-9)" }}>
      <ReadingNav current={read.n} />

      <header style={{ marginBottom: "var(--sp-6)" }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--primary)", fontWeight: "var(--w-semibold)" }}>Read {read.n} of {READS.length}</div>
        <h1 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-display)", fontWeight: "var(--w-semibold)", letterSpacing: "var(--ls-tight)", lineHeight: "var(--lh-tight)", margin: "0.5rem 0 0" }}>About TB</h1>
      </header>

      <ReadSection n={1} title="What is TB?">
        <Prose>
          Tuberculosis (TB) is an infectious disease caused by the bacterium <em>Mycobacterium tuberculosis</em>.
          It primarily attacks the lungs (pulmonary TB), but it can spread to almost any part of the body,
          including the kidneys, spine, and brain.<Cite n={2} /> TB spreads through the air: when a person with
          active lung TB coughs, sneezes, or spits, they expel bacteria that others can inhale — and breathing in
          just a few germs is enough to become infected.<Cite n={2} />
        </Prose>
        <Prose style={{ margin: "0 0 var(--sp-5)" }}>
          There are two forms of the infection, and the difference between them matters for both the patient and
          everyone around them.
        </Prose>
        <CardGrid cols={2}>
          <InfoCard tone="danger" title="Active TB">
            <Bullets items={[
              "The bacteria multiply and make the person sick.",
              <>Symptoms include a chronic cough (often with bloody mucus), chest pain, severe weight loss, night sweats, and fever.</>,
              <>People with active lung TB are <strong>highly contagious</strong>.</>,
            ]} />
          </InfoCard>
          <InfoCard tone="muted" title="Latent TB">
            <Bullets items={[
              "The bacteria are present, but the immune system keeps them from becoming active disease.",
              "No symptoms, the person does not feel sick, and they cannot spread it to others.",
              <>About a quarter of the world's population is estimated to carry TB bacteria; most never develop the disease and some clear it entirely.<Cite n={2} /></>,
            ]} />
          </InfoCard>
        </CardGrid>
      </ReadSection>

      <ReadSection n={2} title="Key properties of TB">
        <CardGrid cols={3}>
          <InfoCard title="Drug resistance">
            <Bullets items={[
              <>Multidrug-resistant TB (MDR-TB) is a growing public-health security threat.<Cite n={4} /></>,
              "When patients don't finish their months-long antibiotics, surviving bacteria mutate and resist standard drugs.",
              <>Treatment then needs expensive, highly toxic medicines that take far longer to work — and only about 2 in 5 people with drug-resistant TB accessed treatment in 2024.<Cite n={1} /></>,
            ]} />
          </InfoCard>
          <InfoCard title="Lethal synergy with HIV">
            <Bullets items={[
              <>TB is the leading cause of death among people living with HIV.<Cite n={3} /></>,
              <>Because HIV weakens the immune system, people with the virus are about <strong>12&times; more likely</strong> to develop active TB.<Cite n={3} /></>,
              "It is also a major contributor to antimicrobial resistance.",
            ]} />
          </InfoCard>
          <InfoCard title="Economic devastation">
            <Bullets items={[
              "TB mostly strikes adults in their most productive working years.",
              <>Globally, about <strong>half</strong> of people treated for TB face &ldquo;catastrophic costs&rdquo; — losing more than 20% of household income to medical bills and lost wages.<Cite n={1} /></>,
              "Over 80% of cases and deaths fall on low- and middle-income countries.",
            ]} />
          </InfoCard>
        </CardGrid>
      </ReadSection>

      <ReadSection n={3} title="TB at a glance">
        <div style={{ border: "1px solid var(--line)", borderRadius: "var(--r)", overflow: "hidden", background: "var(--surface)" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--paper-2)" }}>
                <th style={{ textAlign: "left", padding: "0.7rem 1.1rem", fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--ink-3)", fontWeight: "var(--w-semibold)" }}>Global TB impact (2024)</th>
                <th style={{ textAlign: "left", padding: "0.7rem 1.1rem", fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--ink-3)", fontWeight: "var(--w-semibold)" }}>The numbers</th>
              </tr>
            </thead>
            <tbody>
              <TableRow head="New active cases" first>10.7 million people fell ill (5.8m men, 3.7m women, 1.2m children).<Cite n={1} /></TableRow>
              <TableRow head="Latent infections">Roughly 2 billion people — about a quarter of the global population — carry the dormant bacteria.<Cite n={2} /></TableRow>
              <TableRow head="Lifetime risk">People with latent TB have a 5–10% chance of developing active TB in their lifetime.<Cite n={1} /></TableRow>
              <TableRow head="Deaths">1.23 million people died from TB in 2024 (including 150,000 among people with HIV) — the world's leading cause of death from a single infectious agent.<Cite n={1} /></TableRow>
            </tbody>
          </table>
        </div>
      </ReadSection>

      <ReadSection n={4} title="Key facts">
        <Prose style={{ margin: "0 0 var(--sp-4)" }}>
          The headline numbers behind TB's global burden, from the WHO's 2024 reporting. Expand to read them.
        </Prose>
        <Collapsible title="WHO key facts — 2024" subtitle="Global tuberculosis figures" defaultOpen={false}>
          <Bullets items={[
            <>A total of <strong>1.23 million</strong> people died from TB in 2024 (including 150,000 among people with HIV). TB is the world's leading cause of death from a single infectious agent and among the top 10 causes of death overall.<Cite n={1} /></>,
            <>TB was both the leading killer of people with HIV in 2024 and a major cause of deaths related to antimicrobial resistance.<Cite n={1} /></>,
            <>An estimated <strong>10.7 million</strong> people fell ill with TB worldwide in 2024 — 5.8 million men, 3.7 million women, and 1.2 million children. TB is present in all countries and age groups.<Cite n={1} /></>,
            <>MDR-TB remains a public-health crisis: only about 2 in 5 people with drug-resistant TB accessed treatment in 2024.<Cite n={1} /></>,
            <>Global efforts have saved an estimated <strong>83 million lives</strong> since the year 2000.<Cite n={1} /></>,
            <>TB is preventable and curable.<Cite n={1} /></>,
            <>Over 80% of cases and deaths are in low- and middle-income countries. In 2024 the most new cases were in the WHO South-East Asia Region (34%), then the Western Pacific (27%) and Africa (25%).<Cite n={1} /></>,
            <>Around 87% of new cases occurred in the 30 high-burden countries, with two-thirds of the global total in India (25%), Indonesia (10%), the Philippines (6.8%), China (6.5%), Pakistan (6.3%), Nigeria (4.8%), the Democratic Republic of the Congo (3.9%), and Bangladesh (3.6%). The top five alone accounted for 55%.<Cite n={[1, 5]} /></>,
            <>About 50% of people treated for TB and their households face catastrophic costs. In 2024 an estimated 0.97 million new cases were attributable to undernutrition, 0.93 million to diabetes, 0.74 million to alcohol use disorders, 0.70 million to smoking, and 0.57 million to HIV.<Cite n={1} /></>,
          ]} />
        </Collapsible>
      </ReadSection>

      <SourceList sources={SOURCES} />
      <NextSection current={read.n} />
    </main>
  );
}
