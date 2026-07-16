import React from "react";
import {
  ReviewPage,
  ReviewHeader,
  ReadSection,
  Prose,
  Bullets,
  CardGrid,
  InfoCard,
  KeyIdea,
  Collapsible,
  Bibliography,
  makeCite,
  type Source,
} from "@/components/Reading";
import { Figure, BarChart, StackedBar, Cascade } from "@/components/charts";
import { XrayView, XraySource } from "@/components/XrayFigure";

// All figures on this page use the WHO's 2024 reporting year (published in the
// Global TB Report 2025) — the latest available. Where an older source in the
// project's document set quoted 2022 figures (1.3m deaths / 10.6m cases), the
// newer numbers supersede them throughout.
const SOURCES: Source[] = [
  { label: "WHO — Tuberculosis fact sheet (2024 figures)", url: "https://www.who.int/news-room/fact-sheets/detail/tuberculosis" },
  { label: "WHO — Tuberculosis (health topic)", url: "https://www.who.int/health-topics/tuberculosis" },
  { label: "CDC — Combating global TB", url: "https://www.cdc.gov/global-hiv-tb/php/our-approach/combatingglobaltb.html" },
  { label: "Treatment outcomes of MDR-TB (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11995120/" },
  { label: "WHO Global TB Report 2025 — summary (EATG)", url: "https://www.eatg.org/hiv-news/who-global-tb-report-2025-global-gains-in-tb-response-endangered-by-funding-challenges/" },
  { label: "WHO Global Tuberculosis Report 2025 (ReliefWeb)", url: "https://reliefweb.int/report/world/global-tuberculosis-report-2025" },
  { label: "TBX11K — Rethinking Computer-aided Tuberculosis Diagnosis (Liu et al., CVPR 2020)", url: "https://github.com/yun-liu/Tuberculosis" },
];

const Cite = makeCite(SOURCES);

function TableRow({ head, children, first }: { head: string; children: React.ReactNode; first?: boolean }) {
  return (
    <tr>
      <th
        scope="row"
        style={{
          textAlign: "left",
          verticalAlign: "top",
          width: "34%",
          padding: "0.85rem 1.1rem",
          borderTop: first ? "none" : "1px solid var(--line)",
          fontFamily: "var(--font-sans)",
          fontSize: "var(--text-sm)",
          fontWeight: "var(--w-semibold)",
          color: "var(--ink)",
        }}
      >
        {head}
      </th>
      <td
        style={{
          verticalAlign: "top",
          padding: "0.85rem 1.1rem",
          borderTop: first ? "none" : "1px solid var(--line)",
          fontFamily: "var(--font-sans)",
          fontSize: "var(--text-sm)",
          color: "var(--ink-2)",
          lineHeight: "var(--lh-snug)",
        }}
      >
        {children}
      </td>
    </tr>
  );
}

export function AboutTBPage() {
  return (
    <ReviewPage path="/about-tb">
      <ReviewHeader
        n={1}
        title="About TB"
        lead="Tuberculosis is the world's deadliest single infectious agent, it is curable, and it kills over a million people a year anyway. That gap — between curable and killing — is what this whole project is aimed at."
      />

      <ReadSection
        id="what-is-tb"
        n={1}
        title="What is TB?"
        lead="An airborne bacterial infection with two very different forms — and the difference matters for the patient and for everyone near them."
      >
        <Prose>
          Tuberculosis (TB) is an infectious disease caused by the bacterium <em>Mycobacterium tuberculosis</em>. It
          primarily attacks the lungs (pulmonary TB), but it can spread to almost any part of the body, including the
          kidneys, spine, and brain.<Cite n={2} /> TB spreads through the air: when a person with active lung TB
          coughs, sneezes, or spits, they expel bacteria that others can inhale — and breathing in just a few germs is
          enough to become infected.<Cite n={2} />
        </Prose>
        <Prose style={{ margin: "0 0 var(--sp-5)" }}>
          There are two forms of the infection, and the difference between them matters for both the patient and
          everyone around them.
        </Prose>
        <CardGrid cols={2}>
          <InfoCard tone="danger" title="Active TB">
            <Bullets
              items={[
                "The bacteria multiply and make the person sick.",
                <>
                  Symptoms include a chronic cough (often with bloody mucus), chest pain, severe weight loss, night
                  sweats, and fever.
                </>,
                <>
                  People with active lung TB are <strong>highly contagious</strong>.
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard tone="muted" title="Latent TB">
            <Bullets
              items={[
                "The bacteria are present, but the immune system keeps them from becoming active disease.",
                "No symptoms, the person does not feel sick, and they cannot spread it to others.",
                <>
                  About a quarter of the world's population is estimated to carry TB bacteria; most never develop the
                  disease and some clear it entirely.<Cite n={2} />
                </>,
              ]}
            />
          </InfoCard>
        </CardGrid>
      </ReadSection>

      <ReadSection
        id="cascade"
        n={2}
        title="From two billion to one million"
        lead="Almost everyone who carries TB never gets ill. Understanding that filter is the key to understanding why screening exists at all."
      >
        <Figure
          n="1.1"
          title="The TB cascade, 2024"
          caption={
            <>
              <strong>The bars are log-scaled and the numbers carry the magnitude.</strong> On a linear scale the final
              stage would be 0.06% of the first and vanish entirely, which would read as "nothing happens here" — the
              opposite of the truth. Each step is labelled with its real conversion rate.
            </>
          }
          source={
            <>
              WHO Tuberculosis fact sheet, 2024 figures<Cite n={1} />; latent-infection prevalence per WHO health
              topic<Cite n={2} />.
            </>
          }
          table={{
            head: ["Stage", "People", "Step"],
            rows: [
              ["Latent infection", "~2,000,000,000", "—"],
              ["Active disease (2024)", "10,700,000", "5–10% lifetime risk"],
              ["Deaths (2024)", "1,230,000", "~11.5% of active cases"],
            ],
          }}
        >
          <Cascade
            stages={[
              {
                label: "carry latent TB",
                value: 2_000_000_000,
                display: "~2 billion",
                note: "About a quarter of everyone alive. No symptoms, not contagious, and most will never know.",
              },
              {
                label: "fell ill in 2024",
                value: 10_700_000,
                display: "10.7 million",
                step: "5–10% lifetime risk of progression",
                note: "5.8m men, 3.7m women, 1.2m children. TB is present in every country and age group.",
              },
              {
                label: "died in 2024",
                value: 1_230_000,
                display: "1.23 million",
                step: "~11.5% of active cases — despite TB being curable",
                note: "Including 150,000 among people living with HIV. The world's leading cause of death from a single infectious agent.",
              },
            ]}
          />
        </Figure>

        <KeyIdea>
          Two billion people carry it; ten million get sick; one million die. The whole clinical problem is finding the
          10.7 million inside the 2 billion — which is a needle-in-haystack problem, which is a screening problem, which
          is why chest X-rays and this project exist.
        </KeyIdea>
      </ReadSection>

      <ReadSection
        id="what-it-looks-like"
        n={3}
        title="What TB looks like on a chest X-ray"
        lead="Before any of the modelling makes sense, it helps to see the actual task. Try to find the lesion before you reveal it."
      >
        <Prose>
          These are real chest X-rays from TBX11K, the dataset this project trains on, with the original radiologists'
          annotations.<Cite n={7} /> The boxes are hidden by default deliberately — look first. If the lesions are not
          obvious to you, that is the correct reaction, and it is the entire reason computer-aided detection is a
          seventy-year-old research field rather than a solved problem.
        </Prose>

        <Figure
          n="1.2"
          title="Active TB — what the detector is hunting"
          caption="Left: a single focal lesion, the classic apical presentation. Right: advanced bilateral disease with four separate active lesions. Reveal the boxes to see the radiologist's ground truth."
          source={<XraySource />}
        >
          <CardGrid cols={2}>
            <XrayView slug="tb-single" />
            <XrayView slug="tb-multi" />
          </CardGrid>
        </Figure>

        <Figure
          n="1.3"
          title="The three ways to be wrong"
          caption={
            <>
              These are the confusions that make TB detection hard. <strong>Obsolete TB</strong> is healed scarring — a
              cured patient who needs no treatment, whose tissue nonetheless looks like disease. <strong>Sick non-TB</strong>{" "}
              lungs have real pathology that is not TB; our detector fires on them 2–3× more often than on healthy
              lungs. And <strong>healthy</strong> lungs are the overwhelming majority of any screening queue, so even a
              tiny false-alarm rate produces a flood of unnecessary confirmatory tests.
            </>
          }
          source={<XraySource />}
        >
          <CardGrid cols={3}>
            <XrayView slug="tb-obsolete" height={240} />
            <XrayView slug="sick" height={240} />
            <XrayView slug="healthy" height={240} />
          </CardGrid>
        </Figure>

        <Prose style={{ marginBottom: 0 }}>
          One number puts this in perspective. On this same benchmark, experienced radiologists reached only{" "}
          <strong>68.7% accuracy</strong> against the gold standard.<Cite n={7} /> Any model claiming near-perfect
          accuracy on TB is either solving an easier problem than this one, or measuring itself badly.
        </Prose>
      </ReadSection>

      <ReadSection
        id="properties"
        n={4}
        title="Key properties of TB"
        lead="Three forces that make TB a fundamentally different problem from other chest pathology."
      >
        <CardGrid cols={3}>
          <InfoCard title="Drug resistance">
            <Bullets
              items={[
                <>
                  Multidrug-resistant TB (MDR-TB) is a growing public-health security threat.<Cite n={4} />
                </>,
                "When patients don't finish their months-long antibiotics, surviving bacteria mutate and resist standard drugs.",
                <>
                  Treatment then needs expensive, highly toxic medicines that take far longer to work — and only about
                  2 in 5 people with drug-resistant TB accessed treatment in 2024.<Cite n={1} />
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard title="Lethal synergy with HIV">
            <Bullets
              items={[
                <>
                  TB is the leading cause of death among people living with HIV.<Cite n={3} />
                </>,
                <>
                  Because HIV weakens the immune system, people with the virus are about <strong>12× more likely</strong>{" "}
                  to develop active TB.<Cite n={3} />
                </>,
                "It is also a major contributor to antimicrobial resistance.",
              ]}
            />
          </InfoCard>
          <InfoCard title="Economic devastation">
            <Bullets
              items={[
                "TB mostly strikes adults in their most productive working years.",
                <>
                  Globally, about <strong>half</strong> of people treated for TB face &ldquo;catastrophic costs&rdquo; —
                  losing more than 20% of household income to medical bills and lost wages.<Cite n={1} />
                </>,
                "Over 80% of cases and deaths fall on low- and middle-income countries.",
              ]}
            />
          </InfoCard>
        </CardGrid>
      </ReadSection>

      <ReadSection
        id="geography"
        n={5}
        title="Where TB happens"
        lead="TB is not evenly distributed. It concentrates — and it concentrates exactly where radiologists are scarcest."
      >
        <Figure
          n="1.4"
          title="New TB cases by WHO region, 2024"
          caption={
            <>
              Three regions account for <strong>86%</strong> of all new cases. This is the single most important fact
              about deploying TB AI: the burden sits precisely where trained thoracic radiologists are least available,
              which is why an offline model on a portable X-ray unit can matter more than a marginal accuracy gain.
            </>
          }
          source={
            <>
              WHO Tuberculosis fact sheet, 2024 figures<Cite n={1} />. Remaining regions inferred as the balance to
              100%.
            </>
          }
          table={{
            head: ["WHO region", "Share of new cases"],
            rows: [
              ["South-East Asia", "34%"],
              ["Western Pacific", "27%"],
              ["Africa", "25%"],
              ["Eastern Mediterranean, Americas, Europe", "14%"],
            ],
          }}
        >
          <StackedBar
            segments={[
              { label: "South-East Asia", value: 34, slot: 1 },
              { label: "Western Pacific", value: 27, slot: 2 },
              { label: "Africa", value: 25, slot: 3 },
              { label: "Rest of world", value: 14, slot: 4 },
            ]}
          />
        </Figure>

        <Figure
          n="1.5"
          title="The eight countries carrying two-thirds of global TB, 2024"
          caption={
            <>
              India alone accounts for a quarter of the world's TB. The top five countries together account for 55%,
              and around 87% of new cases occur in just 30 high-burden countries.
            </>
          }
          source={
            <>
              WHO Tuberculosis fact sheet<Cite n={1} /> and Global TB Report 2025 summary<Cite n={5} />.
            </>
          }
          table={{
            head: ["Country", "Share of global cases"],
            rows: [
              ["India", "25%"],
              ["Indonesia", "10%"],
              ["Philippines", "6.8%"],
              ["China", "6.5%"],
              ["Pakistan", "6.3%"],
              ["Nigeria", "4.8%"],
              ["DR Congo", "3.9%"],
              ["Bangladesh", "3.6%"],
            ],
          }}
        >
          <BarChart
            unit="%"
            labelWidth={110}
            bars={[
              { label: "India", value: 25, display: "25%", emphasis: true },
              { label: "Indonesia", value: 10, display: "10%" },
              { label: "Philippines", value: 6.8, display: "6.8%" },
              { label: "China", value: 6.5, display: "6.5%" },
              { label: "Pakistan", value: 6.3, display: "6.3%" },
              { label: "Nigeria", value: 4.8, display: "4.8%" },
              { label: "DR Congo", value: 3.9, display: "3.9%" },
              { label: "Bangladesh", value: 3.6, display: "3.6%" },
            ]}
          />
        </Figure>

        <Figure
          n="1.6"
          title="New TB cases attributable to risk factors, 2024"
          caption={
            <>
              These overlap and do not sum to the total. The striking one is undernutrition: the largest single
              attributable driver of TB worldwide is <em>not having enough food</em>. TB is a disease of poverty with a
              bacterial cause.
            </>
          }
          source={
            <>
              WHO Tuberculosis fact sheet, 2024 figures<Cite n={1} />.
            </>
          }
          table={{
            head: ["Risk factor", "Attributable cases (millions)"],
            rows: [
              ["Undernutrition", "0.97"],
              ["Diabetes", "0.93"],
              ["Alcohol use disorders", "0.74"],
              ["Smoking", "0.70"],
              ["HIV", "0.57"],
            ],
          }}
        >
          <BarChart
            labelWidth={150}
            bars={[
              { label: "Undernutrition", value: 0.97, display: "0.97m", emphasis: true },
              { label: "Diabetes", value: 0.93, display: "0.93m" },
              { label: "Alcohol use disorders", value: 0.74, display: "0.74m" },
              { label: "Smoking", value: 0.7, display: "0.70m" },
              { label: "HIV", value: 0.57, display: "0.57m" },
            ]}
          />
        </Figure>
      </ReadSection>

      <ReadSection id="at-a-glance" n={6} title="TB at a glance" lead="The headline numbers, from the WHO's 2024 reporting year.">
        <div style={{ border: "1px solid var(--line)", borderRadius: "var(--r)", overflow: "hidden", background: "var(--surface)" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--paper-2)" }}>
                {["Global TB impact (2024)", "The numbers"].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: "left",
                      padding: "0.7rem 1.1rem",
                      fontFamily: "var(--font-mono)",
                      fontSize: "var(--text-label)",
                      textTransform: "uppercase",
                      letterSpacing: "var(--ls-label)",
                      color: "var(--ink-3)",
                      fontWeight: "var(--w-semibold)",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <TableRow head="New active cases" first>
                10.7 million people fell ill (5.8m men, 3.7m women, 1.2m children).<Cite n={1} />
              </TableRow>
              <TableRow head="Latent infections">
                Roughly 2 billion people — about a quarter of the global population — carry the dormant bacteria.
                <Cite n={2} />
              </TableRow>
              <TableRow head="Lifetime risk">
                People with latent TB have a 5–10% chance of developing active TB in their lifetime.<Cite n={1} />
              </TableRow>
              <TableRow head="Deaths">
                1.23 million people died from TB in 2024 (including 150,000 among people with HIV) — the world's
                leading cause of death from a single infectious agent.<Cite n={1} />
              </TableRow>
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: "var(--sp-5)" }}>
          <Collapsible title="WHO key facts — 2024" subtitle="The full set of headline figures">
            <Bullets
              items={[
                <>
                  A total of <strong>1.23 million</strong> people died from TB in 2024 (including 150,000 among people
                  with HIV). TB is the world's leading cause of death from a single infectious agent and among the top
                  10 causes of death overall.<Cite n={1} />
                </>,
                <>
                  TB was both the leading killer of people with HIV in 2024 and a major cause of deaths related to
                  antimicrobial resistance.<Cite n={1} />
                </>,
                <>
                  An estimated <strong>10.7 million</strong> people fell ill with TB worldwide in 2024 — 5.8 million
                  men, 3.7 million women, and 1.2 million children. TB is present in all countries and age groups.
                  <Cite n={1} />
                </>,
                <>
                  MDR-TB remains a public-health crisis: only about 2 in 5 people with drug-resistant TB accessed
                  treatment in 2024.<Cite n={1} />
                </>,
                <>
                  Global efforts have saved an estimated <strong>83 million lives</strong> since the year 2000.
                  <Cite n={1} />
                </>,
                <>TB is preventable and curable.<Cite n={1} /></>,
                <>
                  Over 80% of cases and deaths are in low- and middle-income countries. In 2024 the most new cases were
                  in the WHO South-East Asia Region (34%), then the Western Pacific (27%) and Africa (25%).
                  <Cite n={1} />
                </>,
                <>
                  Around 87% of new cases occurred in the 30 high-burden countries, with two-thirds of the global total
                  in India (25%), Indonesia (10%), the Philippines (6.8%), China (6.5%), Pakistan (6.3%), Nigeria
                  (4.8%), the Democratic Republic of the Congo (3.9%), and Bangladesh (3.6%). The top five alone
                  accounted for 55%.<Cite n={[1, 5]} />
                </>,
                <>
                  About 50% of people treated for TB and their households face catastrophic costs. In 2024 an estimated
                  0.97 million new cases were attributable to undernutrition, 0.93 million to diabetes, 0.74 million to
                  alcohol use disorders, 0.70 million to smoking, and 0.57 million to HIV.<Cite n={1} />
                </>,
              ]}
            />
          </Collapsible>
        </div>
      </ReadSection>

      <Bibliography sources={SOURCES} />
    </ReviewPage>
  );
}
