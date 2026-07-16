import { useNavigate } from "react-router-dom";
import { Callout } from "@/design-system";
import { Icon } from "@/components/Icon";
import {
  ReviewPage,
  ReviewHeader,
  ReadSection,
  Prose,
  Bullets,
  CardGrid,
  InfoCard,
  KeyIdea,
  Bibliography,
  makeCite,
  type Source,
} from "@/components/Reading";

const SOURCES: Source[] = [
  { label: "Isoniazid-induced hepatotoxicity (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7043866/" },
  { label: "CDC — Latent tuberculosis infection", url: "https://www.cdc.gov/tb/hcp/clinical-overview/latent-tuberculosis-infection.html" },
  { label: "Xpert MTB/RIF (GeneXpert) diagnostics (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC5556276/" },
  { label: "AAFP — LTBI screening & treatment guidelines", url: "https://www.aafp.org/about/news/20200219ltbiguidelines" },
  { label: "WHO — Chest radiography & CAD screening guidelines (Module 2)", url: "https://www.who.int/publications/i/item/9789240022676" },
  { label: "NCBI Bookshelf — TB diagnosis overview", url: "https://www.ncbi.nlm.nih.gov/books/NBK569341/" },
  { label: "WHO — Tuberculosis fact sheet (2024 figures)", url: "https://www.who.int/news-room/fact-sheets/detail/tuberculosis" },
];

const Cite = makeCite(SOURCES);

function ReadMore() {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate("/active-tb-detection")}
      style={{
        all: "unset",
        boxSizing: "border-box",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "1rem",
        width: "100%",
        marginTop: "var(--sp-5)",
        padding: "var(--sp-5)",
        border: "1px solid var(--primary-tint-2)",
        background: "var(--primary-tint)",
        borderRadius: "var(--r)",
        transition: "box-shadow var(--transition)",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.boxShadow = "none";
      }}
    >
      <span>
        <span
          style={{
            display: "block",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-label)",
            textTransform: "uppercase",
            letterSpacing: "var(--ls-label)",
            color: "var(--primary)",
            fontWeight: "var(--w-semibold)",
            marginBottom: "0.25rem",
          }}
        >
          Read more
        </span>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-body)", fontWeight: "var(--w-semibold)", color: "var(--ink)" }}>
          How is active TB detected — the four methods compared
        </span>
      </span>
      <Icon name="arrow-right" size="1.2rem" style={{ color: "var(--primary)", flexShrink: 0 }} />
    </button>
  );
}

export function HowDetectedPage() {
  return (
    <ReviewPage path="/how-tb-detected">
      <ReviewHeader
        n={2}
        title="How is TB detected?"
        lead={
          <>
            Detection splits in two: finding <strong>latent</strong> infection versus <strong>active</strong> disease.
            The first question, though, is why we screen at all instead of simply treating everyone.
          </>
        }
      />

      <ReadSection
        id="why-not-treat"
        n={1}
        title="Why don't we treat everyone?"
        lead="Two billion people carry TB. Treating them all would be a catastrophe — here is the arithmetic."
      >
        <Prose>
          Roughly 2 billion people carry latent TB, but only 5–10% ever progress to active disease.<Cite n={7} />{" "}
          Treating the entire infected population would mean giving harsh drugs to about 1.8 billion people who would
          never have fallen ill — and that does real harm.
        </Prose>
        <CardGrid cols={2}>
          <InfoCard title="Hepatotoxicity (liver damage)">
            <Bullets
              items={[
                <>
                  The first-line drugs for latent TB (isoniazid, rifampin) are harsh and carry a real risk of
                  drug-induced liver injury.<Cite n={1} />
                </>,
                "Given globally, they would cause millions of cases of severe liver damage — far outweighing the preventive benefit.",
              ]}
            />
          </InfoCard>
          <InfoCard title="Adherence & resistance threat">
            <Bullets
              items={[
                "Latent-TB treatment runs 3 to 9 months, and patients who feel perfectly healthy rarely finish long, toxic regimens.",
                "Widespread partial treatment would rapidly breed multidrug-resistant TB — a far deadlier crisis.",
              ]}
            />
          </InfoCard>
        </CardGrid>
        <Callout kind="note" title="Who does get treated" style={{ marginTop: "var(--sp-4)" }}>
          Because of these risks, the WHO and CDC recommend treating latent TB only in high-risk groups — people whose
          immune systems are compromised (such as those living with HIV or organ-transplant recipients), or anyone
          recently in close contact with an active TB patient.<Cite n={2} />
        </Callout>

        <KeyIdea>
          Because we cannot treat everyone, we must find the few. Screening is not a convenience — it is the only
          ethical option available, and its accuracy is the whole ballgame.
        </KeyIdea>
      </ReadSection>

      <ReadSection
        id="latent"
        n={2}
        title="Detecting latent TB"
        lead="No symptoms, no bacteria in the sputum. Both tests look for the immune system's memory instead."
      >
        <Prose>
          Latent TB hides — there are no symptoms and no bacteria in the sputum to find. Instead, both tests look for
          the immune system's <em>memory</em> of having fought TB.
        </Prose>
        <CardGrid cols={2}>
          <InfoCard title="IGRA (interferon-gamma release assay)">
            <Bullets
              items={[
                "The modern standard blood test. Patient blood is mixed with synthetic TB proteins.",
                "If the immune cells have battled TB before, they rapidly release interferon-gamma, which the assay detects.",
                <>
                  High specificity — it does not trigger false positives from a previous BCG (TB) vaccination.
                  <Cite n={2} />
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard title="TST (tuberculin skin test / Mantoux)">
            <Bullets
              items={[
                "A small amount of TB protein is injected just under the skin of the arm.",
                "If the body recognises it, a hard raised bump forms after 48–72 hours.",
                <>
                  Cheaper but less accurate than IGRA — it often reads false-positive in people given the BCG vaccine
                  as children.<Cite n={4} />
                </>,
              ]}
            />
          </InfoCard>
        </CardGrid>
      </ReadSection>

      <ReadSection
        id="active"
        n={3}
        title="Detecting active TB"
        lead="Now the bacteria are multiplying and shedding — so they can be found directly. The methods trade off cost, speed, and certainty."
      >
        <Prose>
          Active TB can be found directly, because the bacteria are multiplying and shedding into the lungs and sputum.
          The methods trade off cost, speed, and certainty.
        </Prose>
        <CardGrid cols={2}>
          <InfoCard title="Rapid molecular assays (e.g. GeneXpert)">
            <Bullets
              items={[
                "The modern frontline tool. An automated PCR test amplifies and detects M. tuberculosis DNA from a sputum sample.",
                <>
                  Crucially, it detects Rifampicin-resistance mutations at the same time — in under two hours.
                  <Cite n={3} />
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard title="Sputum smear microscopy">
            <Bullets
              items={[
                "A century-old method still widely used in low-resource settings: sputum is stained and examined for acid-fast bacilli.",
                <>Cheap, but low sensitivity — it misses many cases.<Cite n={4} /></>,
              ]}
            />
          </InfoCard>
          <InfoCard title="Mycobacterial culture">
            <Bullets
              items={[
                "The gold standard for confirmation: sputum is grown in a nutrient-rich medium.",
                "Highly accurate and allows full drug-susceptibility testing — but TB grows slowly, so results take 2–6 weeks.",
              ]}
            />
          </InfoCard>
          <InfoCard tone="primary" title="Radiology (chest X-ray)">
            <Bullets
              items={[
                "Reveals the structural damage, cavities, and fluid build-up caused by active infection.",
                <>
                  Fast and cheap enough to run at scale, which makes it the frontline screening filter — increasingly
                  read by AI/CAD software.<Cite n={[5, 6]} />
                </>,
              ]}
            />
          </InfoCard>
        </CardGrid>
        <Prose style={{ marginTop: "var(--sp-5)" }}>
          Chest X-rays are the perfect frontline filter precisely because they are fast and inexpensive. If every person
          with a bad cough needed a $15–$20 GeneXpert test, high-burden health systems would collapse under the cost and
          backlog. Highly accurate AI and CAD (computer-aided detection) software lets a clinic screen thousands of
          X-rays quickly — safely sending home the clear lungs and reserving the expensive lab tests for the images
          flagged as high-risk.<Cite n={5} />
        </Prose>
        <ReadMore />
      </ReadSection>

      <Bibliography sources={SOURCES} />
    </ReviewPage>
  );
}
