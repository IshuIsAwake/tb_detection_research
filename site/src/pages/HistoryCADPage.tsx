import { Callout } from "@/design-system";
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
import { Figure, Timeline, BarChart, type Era } from "@/components/charts";

// Deduped: the two source documents cite many of the same works under different
// numbers, so this list is merged by URL and renumbered for this section only.
const SOURCES: Source[] = [
  { label: "The History of Tuberculosis Diagnostics — Co-Dx", url: "https://co-dx.com/history-of-tb-diagnostics/" },
  { label: "The long and winding road of chest radiography for tuberculosis detection — European Respiratory Journal", url: "https://publications.ersnet.org/content/erj/49/5/1700364" },
  { label: "How TB Diagnostics Have Evolved Since the Second Century — ASM", url: "https://asm.org/articles/2021/march/how-tb-diagnostics-have-evolved-since-the-second-c" },
  { label: "Anniversary Paper: History and status of CAD and quantitative image analysis — Medical Physics (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC2673617/" },
  { label: "Computer-aided diagnosis — Wikipedia", url: "https://en.wikipedia.org/wiki/Computer-aided_diagnosis" },
  { label: "Automatic screening for tuberculosis in chest radiographs: a survey — Jaeger et al., QIMS", url: "https://qims.amegroups.org/article/view/1813/html" },
  { label: "Tuberculosis: mother of thoracic surgery then and now — Journal of Thoracic Disease", url: "https://jtd.amegroups.org/article/view/21517/html" },
  { label: "History of computed tomography — Wikipedia", url: "https://en.wikipedia.org/wiki/History_of_computed_tomography" },
  { label: "How CT happened: the early development of medical computed tomography (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8555965/" },
  { label: "Milestones in CT: Past, Present, and Future — Radiology (RSNA)", url: "https://pubs.rsna.org/doi/10.1148/radiol.230803" },
  { label: "Detection and labeling ribs on expiration chest radiographs — Toriwaki et al. (ResearchGate)", url: "https://www.researchgate.net/publication/229076252_Detection_and_labeling_ribs_on_expiration_chest_radiographs" },
  { label: "Spatial Frequency Structure of Chest-photofluorograms — J-Stage", url: "https://www.jstage.jst.go.jp/article/itej1954/29/9/29_9_721/_article/-char/en" },
  { label: "Pulmonary nodules: Computer-aided detection in digital chest images — Giger, Doi & MacMahon (ResearchGate)", url: "https://www.researchgate.net/publication/20858779_Pulmonary_nodules_Computer-aided_detection_in_digital_chest_images" },
  { label: "Eliminating Rib Shadows in Chest Radiographic Images Providing Diagnostic Assistance (ResearchGate)", url: "https://www.researchgate.net/publication/288056407_Eliminating_Rib_Shadows_in_Chest_Radiographic_Images_Providing_Diagnostic_Assistance" },
  { label: "Computer-aided diagnosis in chest radiography: a survey — Pure (TU/e)", url: "https://pure.tue.nl/ws/files/3264740/Metis251436.pdf" },
  { label: "Developments in computer aided diagnosis used for tuberculosis detection using chest radiography — ARPN JEAS", url: "http://www.arpnjournals.org/jeas/research_papers/rp_2016/jeas_0516_4147.pdf" },
  { label: "Computer Aided Tuberculosis Detection, A Review — AIJR Books", url: "https://books.aijr.org/index.php/press/catalog/download/114/44/1494-1?inline=1" },
  { label: "Automatic detection of abnormalities in chest radiographs using local texture analysis — van Ginneken et al. (PubMed)", url: "https://pubmed.ncbi.nlm.nih.gov/11929101/" },
  { label: "Computer-aided diagnosis in chest radiography — van Ginneken, Medical Physics 2001 (Wiley)", url: "https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.1376645" },
  { label: "Bram van Ginneken — Diagnostic Image Analysis Group", url: "https://www.diagnijmegen.nl/people/bram-van-ginneken/" },
  { label: "Chest x-ray analysis — Diagnostic Image Analysis Group, Radboud", url: "https://www.diagnijmegen.nl/research/cxr/" },
  { label: "Diagnostic Accuracy of Computer-Aided Detection of Pulmonary Tuberculosis in Chest Radiographs: A Validation Study from Sub-Saharan Africa (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC4156349/" },
  { label: "Machine and Deep Learning for Tuberculosis Detection on Chest X-Rays: Systematic Literature Review — JMIR", url: "https://www.jmir.org/2023/1/e43154/" },
  { label: "Deep learning-based automatic detection of tuberculosis disease in chest X-ray images (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8906182/" },
  { label: "A novel approach for tuberculosis screening based on deep convolutional neural networks — Hwang et al., SPIE 2016 (Semantic Scholar)", url: "https://www.semanticscholar.org/paper/A-novel-approach-for-tuberculosis-screening-based-Hwang-Kim/212388cada6b67f0937eedc90f99d4fd70ad612f" },
  { label: "Deep Learning at Chest Radiography: Automated Classification of Pulmonary Tuberculosis by Using CNNs — Lakhani & Sundaram, Radiology 2017 (RSNA)", url: "https://pubs.rsna.org/doi/abs/10.1148/radiol.2017162326" },
  { label: "A Systematic Review of Deep Learning Techniques for Tuberculosis Detection From Chest Radiograph — Frontiers in Medicine", url: "https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2022.830515/full" },
  { label: "Deep Learning Algorithms with Demographic Information Help to Detect Tuberculosis in Chest Radiographs — Heo et al. (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6352082/" },
  { label: "Computer aided detection of tuberculosis on chest radiographs: An evaluation of the CAD4TB v6 system (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7099074/" },
  { label: "Comparing different versions of computer-aided detection products when reading chest X-rays for tuberculosis — PLOS Digital Health", url: "https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000067" },
  { label: "Use of computer-aided detection software for tuberculosis screening — WHO policy statement", url: "https://www.who.int/publications/i/item/9789240110373" },
  { label: "Using artificial intelligence to read chest radiographs for tuberculosis detection: a multi-site evaluation (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6802077/" },
  { label: "Computer-aided detection (CAD) for TB — FIND", url: "https://www.finddx.org/what-we-do/projects/computer-aided-detection-cad-for-tb/" },
  { label: "Computer-aided reading of tuberculosis chest radiography: moving the research agenda forward to inform policy — ERS", url: "https://publications.ersnet.org/content/erj/50/1/1700953" },
  { label: "Revisiting Computer-Aided Tuberculosis Diagnosis — Liu et al., TPAMI 2023 (PubMed)", url: "https://pubmed.ncbi.nlm.nih.gov/37934644/" },
];

const Cite = makeCite(SOURCES);

const ERAS: Era[] = [
  {
    id: "antiquity",
    years: "2nd c. CE",
    title: "Burning sputum on coals",
    blurb: "Soranus of Ephesus infers TB from the smell. Diagnosis is symptom-watching: haemoptysis, fever, night sweats — and by then it is late.",
  },
  {
    id: "koch",
    years: "1882",
    title: "Koch finds the bacillus",
    blurb: "On 24 March, Robert Koch presents Mycobacterium tuberculosis. TB becomes a provable infection, not a constitution. Nobel Prize, 1905.",
  },
  {
    id: "mass",
    years: "1930s–70s",
    title: "Mass mobile radiography",
    blurb: "Photofluorographic units in buses, trains and boats screen whole populations. It works — and it is ruinously expensive, and it needs armies of readers.",
  },
  {
    id: "who74",
    years: "1974",
    title: "The WHO calls it off",
    pivot: true,
    blurb: "Mass radiographic case-finding is formally abandoned: too costly, too subjective, too variable between readers. The gap this leaves is precisely the case for automation.",
  },
  {
    id: "aiscr",
    years: "1970s",
    title: "AISCR-V2 — total automation attempted",
    blurb: "Toriwaki, Suenaga and Fukumura build a rule-based system to suppress the ribs and find abnormal shadows. The goal is to replace the radiologist outright.",
  },
  {
    id: "retreat",
    years: "1980s",
    title: "The retreat to 'assistance'",
    pivot: true,
    blurb: "Hand-coding the variability of human anatomy proves computationally intractable. The computer is demoted to a 'second reader' that flags regions — the human decides. CAD is born out of this failure.",
  },
  {
    id: "diffimage",
    years: "1988–90s",
    title: "The difference-image trick",
    blurb: "Giger, Doi and MacMahon subtract a signal-suppressed image from a signal-enhanced one, cancelling the ribcage and leaving nodules as bright islands. ~70% true-positive on isolated nodules.",
  },
  {
    id: "jsrt",
    years: "1998",
    title: "JSRT — the first public benchmark",
    blurb: "247 radiographs released openly. For the first time two algorithms can be compared on the same images.",
  },
  {
    id: "vg",
    years: "2001",
    title: "van Ginneken's texture prototype",
    blurb: "The doctoral work behind CAD4TB: stop hunting geometric nodules, measure diffuse texture instead, and exploit lung symmetry. Outputs a 0–100 abnormality score.",
  },
  {
    id: "nlm",
    years: "~2014",
    title: "Montgomery & Shenzhen released",
    blurb: "The US National Library of Medicine publishes 138 + 662 annotated TB radiographs. The data bottleneck cracks open.",
  },
  {
    id: "lakhani",
    years: "2017",
    title: "Lakhani & Sundaram — AUC 0.99",
    pivot: true,
    blurb: "An AlexNet + GoogLeNet ensemble on 1,007 radiographs reaches AUC 0.99. With a radiologist breaking the 13 ties: 97.3% sensitivity, 100% specificity. The argument is over.",
  },
  {
    id: "cad4tb6",
    years: "2018",
    title: "CAD4TB v6 goes deep",
    blurb: "Delft abandons fifteen years of feature engineering and rebuilds on CNNs. v6 → v7 lifts AUC from 0.823 to 0.903.",
  },
  {
    id: "who21",
    years: "2021 → 2025",
    title: "The WHO endorses CAD — autonomously",
    pivot: true,
    blurb: "CAD is recommended as an alternative to a human reader for people aged 15+. Forty-seven years after 1974, the machine is trusted to screen alone — the 1970s ambition, restored.",
  },
];

export function HistoryCADPage() {
  return (
    <ReviewPage path="/history-cad">
      <ReviewHeader
        n={4}
        title="A history of TB computer-aided diagnosis"
        lead={
          <>
            Teaching a computer to read a chest X-ray is a seventy-year-old project. It is worth knowing that history
            for one reason: this field has already tried the obvious things, and the record of what defeated them is
            the reason today's methods look the way they do.
          </>
        }
      />

      <ReadSection
        id="why-automate"
        n={1}
        title="Why automate at all?"
        lead="Every idea in this section exists because of one stubborn fact: humans read chest X-rays inconsistently."
      >
        <Prose>
          Chest radiography has been the frontline TB imaging tool since the early twentieth century, and it is
          genuinely good at showing the disease — apical infiltrates, reticulonodular opacities, cavitation, calcified
          granulomas, pleural effusion.<Cite n={2} /> The problem was never the image. It was the reading of it.
        </Prose>
        <Prose>
          The thorax is a stack of overlapping structures: ribs, clavicles, and mediastinum superimposed over the lung
          fields. Normal anatomy routinely camouflages early TB.<Cite n={6} /> Add reader fatigue, the numbing effect
          of a screening queue that is overwhelmingly normal, and a severe shortage of thoracic radiologists exactly
          where TB burden is highest, and you get substantial variability both between readers and within the same
          reader on different days.<Cite n={[4, 6]} />
        </Prose>

        <Callout kind="note" title="How hard is it really?">
          Not a rhetorical question. On the TBX11K benchmark this project trains on, experienced radiologists scored{" "}
          <strong>68.7% accuracy</strong> against the gold standard.<Cite n={35} /> That number is the honest baseline
          for everything that follows — including our own detector.
        </Callout>

        <Prose style={{ marginBottom: 0 }}>
          The mass-radiography campaigns of the mid-century made this concrete. Mobile photofluorographic units — the
          Odelca camera in buses, trains and boats — screened whole populations and did find enormous numbers of
          cases.<Cite n={2} /> They were also ruinously expensive and demanded that skilled physicians review thousands
          of miniature films a day. As TB prevalence fell in industrialised nations, the economics collapsed, and in{" "}
          <strong>1974 the WHO formally recommended abandoning</strong> indiscriminate mass radiographic
          case-finding.<Cite n={2} />
        </Prose>

        <KeyIdea>
          1974 is the hinge. If human reading is too costly and too variable to screen at scale, then high-throughput
          triage only comes back if a machine can do the reading. The entire field is an answer to a gap the WHO
          created.
        </KeyIdea>
      </ReadSection>

      <ReadSection
        id="timeline"
        n={2}
        title="Seventy years in one column"
        lead="The whole arc, before we walk through it. The highlighted moments are the times the field reversed itself."
      >
        <Figure
          n="4.1"
          title="From burning sputum to autonomous AI screening"
          caption={
            <>
              Highlighted entries are pivots — moments the field changed its mind about what computers were <em>for</em>.
              Notice that there are three, and that the last one undoes the second.
            </>
          }
        >
          <Timeline eras={ERAS} />
        </Figure>
      </ReadSection>

      <ReadSection
        id="genesis"
        n={3}
        title="The genesis: total automation, and its collapse"
        lead="The 1970s tried to build a diagnostic robot. Understanding why that failed explains the next forty years."
      >
        <Prose>
          The theoretical groundwork was laid in the mid-1950s, when Lee Lusted began asking whether early computers
          could analyse radiographic abnormalities.<Cite n={4} /> What made it plausible was CT. Cormack's 1963
          proposal and Hounsfield's 1968 prototype — and the first clinical brain scan in 1971, an 80×80 image that
          took hours to reconstruct — proved that computers could extract radiological information beyond the naked
          eye.<Cite n={[8, 9, 10]} />
        </Prose>
        <Prose>
          In TB specifically, the landmark was the <strong>AISCR-V2</strong> system (and its successor V3) built by
          Jun-ichiro Toriwaki, Yasuhito Suenaga and Teruo Fukumura.<Cite n={11} /> Its designers understood the real
          geometric problem: you cannot find a subtle lesion until you have found and suppressed the ribs on top of it.
          So AISCR-V2 modelled an artificial "expiration lung field" using constructs like a hemi-elliptical cavity to
          make rib edges computable, ran edge-detection operators with hand-coded rules — "4-way with 10-neighbors
          connectivity" — to map clavicles and rib margins, and only then hunted abnormal shadows in the intercostal
          spaces.<Cite n={[11, 12]} />
        </Prose>
        <Prose>
          The stated goal was to <em>fully automate the chest exam</em> and replace the radiologist.<Cite n={4} /> It
          did not survive contact with the 1970s. Film digitisation was noisy and low-resolution, compute was scarce —
          but the fatal problem was mathematical. Work by Richard Karp on combinatorial complexity helped the medical
          informatics community realise that hand-coding a flowchart to cover the near-infinite variability of human
          anatomy, positioning and pathology was computationally intractable.<Cite n={5} />
        </Prose>

        <CardGrid cols={2}>
          <InfoCard tone="muted" title="The ambition (1960s–70s)">
            <Bullets
              items={[
                "Goal: a diagnostic robot that reads the film and issues the diagnosis.",
                "Method: explicit geometric rules, hand-written by humans.",
                <>
                  Outcome: defeated by biological variability. The rules could never cover the cases.<Cite n={5} />
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard tone="primary" title="The compromise (1980s onward)">
            <Bullets
              items={[
                <>
                  Goal: a <strong>second reader</strong> that flags suspicious regions.<Cite n={4} />
                </>,
                "Method: the computer detects mathematical anomalies; the human characterises and decides.",
                "Outcome: clinically useful, and the field's self-image for the next forty years.",
              ]}
            />
          </InfoCard>
        </CardGrid>
      </ReadSection>

      <ReadSection
        id="heuristic"
        n={4}
        title="Heuristics: subtracting the ribcage"
        lead="The 1980s and 90s got clever about the anatomy problem without ever solving the generalisation problem."
      >
        <Prose>
          Digitisation made real image processing possible. Early 1990s laser drum scanners turned film into 512×512
          matrices with 10-bit converters; Computed and then Digital Radiography removed film from the loop
          entirely.<Cite n={4} /> That enabled the enhancement toolkit still in use today — homomorphic filtering to
          even out illumination, and <strong>CLAHE</strong> to lift contrast at the lung apices, where early TB likes
          to sit, without amplifying noise across the whole image.<Cite n={16} />
        </Prose>
        <Prose>
          The most elegant idea of the era came from Giger, Doi and MacMahon at the University of Chicago: the{" "}
          <strong>difference-image approach</strong>.<Cite n={13} /> Rather than fight the ribcage, cancel it. Build
          one image with a filter matched to a nodule's profile, so nodules brighten. Build a second with a ring filter
          that replaces each pixel with the average around a radius — nodules are smaller than the ring, so they blur
          away while ribs and vessels survive. Subtract the second from the first and the anatomy largely cancels,
          leaving nodules as isolated bright islands.<Cite n={13} />
        </Prose>
        <Prose>
          Surviving islands were then filtered by explicit hand-written tests: a <em>growth test</em> (true nodules
          grow concentrically as the threshold drops; vessels suddenly merge), a <em>circularity profile</em> (nodules
          are round, vessels are not), and a <em>slope test</em> on the intensity gradient.<Cite n={13} /> It reached
          roughly 70% true-positive on isolated nodules — and there it stopped. The pipeline was built for round
          things, and active TB is frequently not round: patchy consolidation, miliary mottle, irregular
          cavities.<Cite n={2} />
        </Prose>
      </ReadSection>

      <ReadSection
        id="classical-ml"
        n={5}
        title="Classical machine learning — and the birth of CAD4TB"
        lead="The 2000s replaced hand-written rules with hand-written features, and learned the classifier instead."
      >
        <Prose>
          The 2000s pipeline had four fixed stages: <strong>segmentation → feature extraction → feature selection →
          classification</strong>.<Cite n={16} /> Lungs were isolated with graph-cut optimisation or Active Shape
          Models — a statistical template of lung shape, iteratively deformed until it snapped to the patient's own rib
          cage and diaphragm.<Cite n={[15, 16]} /> Then, because TB presents as amorphous cloudiness rather than
          geometry, the field pivoted from shape to <strong>texture</strong>: first-order histogram statistics
          (variance, entropy, skewness, kurtosis), Local Binary Patterns, and multiscale Gabor filter banks to quantify
          "fuzziness".<Cite n={[16, 17]} /> Those vectors fed SVMs, k-NN or Random Forests.<Cite n={16} />
        </Prose>
        <Prose>
          Neural networks existed here, but only as end-stage classifiers on low-dimensional vectors a human had
          already computed — sometimes mixed with clinical data such as age and haemoglobin.<Cite n={28} /> They could
          not touch raw pixels.
        </Prose>

        <Callout kind="note" title="CAD4TB: the most-deployed TB AI in the world started here">
          Its foundation was Bram van Ginneken's 2001 doctoral work at Utrecht.<Cite n={[19, 20]} /> The design
          decisions still read as modern: abandon geometric nodule-hunting for diffuse texture; segment with Active
          Shape Models; subdivide the lungs into regional zones; extract multiscale filter-bank moments — and then the
          clever part, <strong>difference features</strong>, subtracting each right-lung region's feature vector from
          its left-lung mirror. Healthy lungs are broadly symmetric; TB is not.<Cite n={19} /> A k-NN classifier with
          leave-one-out training aggregated the zones into a single 0–100 abnormality score.<Cite n={19} />
        </Callout>

        <Prose>
          Grant funding in 2007 turned the prototype into a product — a collaboration between Radboud's Diagnostic
          Image Analysis Group, Delft Imaging Systems, and NGOs in high-burden regions.<Cite n={21} /> Beta v0.01 field
          tested in 2010; v1.08 shipped 2011; v4.10 earned the CE mark.<Cite n={21} /> Field validation in sub-Saharan
          Africa found it could separate confirmed TB from healthy controls with remarkable consistency, frequently
          outperforming mid-level clinical officers — but its specificity stayed measurably below expert thoracic
          radiologists.<Cite n={22} />
        </Prose>

        <KeyIdea>
          Classical ML hit a ceiling that was not about compute. Hand-crafted features are bounded by the vocabulary of
          the humans who wrote them — you cannot measure a pattern you have not thought to name.
        </KeyIdea>
      </ReadSection>

      <ReadSection
        id="data"
        n={6}
        title="The data catalyst"
        lead="Algorithms are downstream of data. Nothing moved until the images became public."
      >
        <Prose>
          Through the 1990s and 2000s, CAD was developed on small proprietary datasets held by individual
          hospitals.<Cite n={2} /> Nobody could benchmark anyone else's claims, verify accuracy, or rule out that a
          model had simply overfitted to one X-ray machine's calibration.<Cite n={2} /> Opening the data was the
          precondition for everything after.
        </Prose>

        <Figure
          n="4.2"
          title="The datasets that unlocked the field"
          caption={
            <>
              JSRT made segmentation benchmarkable; Montgomery and Shenzhen made TB itself benchmarkable. Note the
              scale: the two TB sets together are under 800 images — which is precisely why transfer learning became
              mandatory rather than optional. TBX11K, released in 2020 and used by this project, is more than an order
              of magnitude larger and the first with bounding boxes.
            </>
          }
          source={
            <>
              Sizes from the dataset releases; see <Cite n={6} /> and <Cite n={35} />.
            </>
          }
          table={{
            head: ["Dataset", "Year", "Images", "Contribution"],
            rows: [
              ["JSRT", "1998", 247, "First open segmentation / nodule benchmark"],
              ["Montgomery County", "~2014", 138, "Expert lung masks + TB labels"],
              ["Shenzhen", "~2014", 662, "Scale for early transfer learning"],
              ["TBX11K", "2020", 11200, "Bounding boxes; 5 categories"],
            ],
          }}
        >
          <BarChart
            bars={[
              { label: "JSRT (1998)", value: 247, display: "247" },
              { label: "Montgomery (~2014)", value: 138, display: "138" },
              { label: "Shenzhen (~2014)", value: 662, display: "662" },
              { label: "TBX11K (2020)", value: 11200, display: "11,200", emphasis: true, note: "this project's training set" },
            ]}
            labelWidth={150}
          />
        </Figure>

        <Prose style={{ marginBottom: 0 }}>
          JSRT came first, in 1998: 247 radiographs at 2048×2048 and 12-bit depth, the first standardised open
          benchmark for lung segmentation and nodule detection.<Cite n={6} /> The decisive move came in the early
          2010s, when the US National Library of Medicine released <strong>Montgomery County</strong> (138 images, 58
          TB-positive, with expert lung-field annotations) and <strong>Shenzhen</strong> (662 images, 336
          microbiologically confirmed TB).<Cite n={[6, 24]} /> Fewer than a thousand images between them — trivial by
          ImageNet standards, and enough to change everything.
        </Prose>
      </ReadSection>

      <ReadSection
        id="watershed"
        n={7}
        title="The watershed: 2014–2017"
        lead="Deep learning did not improve the classical pipeline. It deleted it."
      >
        <Prose>
          After ImageNet, the logic of CAD inverted.<Cite n={16} /> Instead of a human defining "compute the local
          binary pattern of this opacity", a deep CNN performs <strong>representation learning</strong>: raw pixels in,
          hierarchical features discovered on the way through — edges, then textures, then disease-specific
          patterns.<Cite n={16} /> The machine was no longer confined to the human radiological vocabulary. Some of
          what it learns maps onto no named sign at all, and correlates with disease anyway.<Cite n={16} />
        </Prose>
        <Prose>
          One problem stood in the way immediately: deep networks want millions of images and TB had fewer than a
          thousand.<Cite n={24} /> The workaround was <strong>transfer learning</strong> — take a network pretrained on
          1.2 million everyday photographs, keep its general-purpose edge and texture filters, and fine-tune the deeper
          layers on the small X-ray set.<Cite n={24} /> Hwang et al. demonstrated it at SPIE in early 2016, reaching
          AUC 0.88–0.96 across three real-world field datasets.<Cite n={25} />
        </Prose>

        <Callout kind="decision" title="Lakhani & Sundaram, Radiology 2017 — the argument ends">
          1,007 posteroanterior radiographs; AlexNet and GoogLeNet, pretrained and augmented, ensembled together. The
          result was unprecedented: <strong>AUC 0.99</strong>, 95.3% accuracy, 92.0% sensitivity, 98.7%
          specificity.<Cite n={26} /> They then simulated a real workflow — on the 13 of 150 cases where the networks
          disagreed, a blinded cardiothoracic radiologist broke the tie. All 13 resolved correctly, pushing the
          ensemble to <strong>97.3% sensitivity and 100% specificity</strong>.<Cite n={26} /> Deep learning could match
          or beat human performance, and human-plus-machine beat either alone.
        </Callout>

        <Prose>
          The field diversified fast — AlexNet gave way to VGG-16/19, ResNet-50 and DenseNet-121, routinely clearing
          90% on accuracy, sensitivity and specificity.<Cite n={24} /> Heo et al. showed that feeding demographics
          (age, weight, height, sex) alongside pixels — "D-CNN" models — stopped sensitivity collapsing at high
          specificity thresholds, proving the machine benefits from the same context a clinician uses.<Cite n={28} />
        </Prose>
        <Prose style={{ marginBottom: 0 }}>
          Commercial software had to follow or die. In 2018 Delft threw away CAD4TB's classical architecture and
          rebuilt it on deep learning as <strong>version 6</strong>.<Cite n={29} /> Independent evaluation against
          microbiological references found the deep versions vastly outperformed every heuristic predecessor and
          matched expert human observers.<Cite n={[29, 30]} />
        </Prose>
      </ReadSection>

      <ReadSection
        id="synthesis"
        n={8}
        title="What the history is actually telling us"
        lead="Three patterns, and one of them is a warning about our own numbers."
      >
        <CardGrid cols={3}>
          <InfoCard tone="primary" title="1 · The pendulum came back">
            <Bullets
              items={[
                "The 1960s wanted full automation. The 1980s gave up and settled for a 'second reader'.",
                <>
                  Deep learning swung it back: since 2021 the WHO endorses CAD as an <strong>alternative to a human
                  reader</strong> for ages 15+.<Cite n={31} />
                </>,
                "The forty-year retreat was a hardware and mathematics constraint, not a truth about medicine.",
              ]}
            />
          </InfoCard>
          <InfoCard title="2 · Algorithms are downstream of data">
            <Bullets
              items={[
                "The difference-image era needed film→digital detectors.",
                <>
                  The deep-learning era needed GPUs <em>and</em> JSRT, Montgomery and Shenzhen at the same
                  time.<Cite n={24} />
                </>,
                "Innovation accelerated the moment data went public — silos, not ideas, were the bottleneck.",
              ]}
            />
          </InfoCard>
          <InfoCard title="3 · Explicit → implicit knowledge">
            <Bullets
              items={[
                <>
                  Rule-based and classical ML both ran on <strong>explicit</strong> human knowledge: circularity, edge
                  gradients, Gabor frequencies.<Cite n={13} />
                </>,
                "That capped the machine at the ceiling of human radiological vocabulary and perceptual bias.",
                "CNNs learn implicitly, straight from pixel distributions — including patterns nobody has named.",
              ]}
            />
          </InfoCard>
        </CardGrid>

        <Callout kind="landmine" title="Read the 99% results with the history in mind" style={{ marginTop: "var(--sp-5)" }}>
          The modern literature is full of 99%+ accuracies. The historical record suggests caution: many come from
          private or unspecified datasets, on binary classification, with no external validation — the exact conditions
          that made pre-2010 accuracy claims unverifiable.<Cite n={23} /> Radiologists score 68.7% on
          TBX11K.<Cite n={35} /> When a paper reports 99.9% on a task where trained humans reach 69%, the honest first
          question is what the benchmark is measuring, not how good the model is. This is the reason
          this project decides on a sealed test set and reports multi-seed bands rather than a best run.
        </Callout>
      </ReadSection>

      <Bibliography sources={SOURCES} />
    </ReviewPage>
  );
}
