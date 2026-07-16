import React from "react";
import { Callout, Badge } from "@/design-system";
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
import { Figure, BarChart, DataTable } from "@/components/charts";
import {
  CNNDiagram,
  ResNetDiagram,
  UNetDiagram,
  DetectorDiagram,
  AttentionDiagram,
  MambaDiagram,
} from "@/components/diagrams";
import { ARCH_ROWS, FOUNDATION_DATASETS, type Family } from "@/data/architectures";

const SOURCES: Source[] = [
  { label: "Deep Learning at Chest Radiography: Automated Classification of Pulmonary Tuberculosis by Using CNNs — Lakhani & Sundaram, Radiology 2017 (RSNA)", url: "https://pubs.rsna.org/doi/abs/10.1148/radiol.2017162326" },
  { label: "Machine and Deep Learning for Tuberculosis Detection on Chest X-Rays: Systematic Literature Review (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10365622/" },
  { label: "Deep learning-based automatic detection of tuberculosis disease in chest X-ray images (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8906182/" },
  { label: "Improving Tuberculosis Detection in Chest X-Ray Images Through Transfer Learning and Deep Learning: Comparative Study of CNN Architectures — JMIRx Med", url: "https://xmed.jmir.org/2025/1/e66029" },
  { label: "A Systematic Review of Deep Learning Techniques for Tuberculosis Detection From Chest Radiograph — Frontiers in Medicine", url: "https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2022.830515/full" },
  { label: "Automatic screening for tuberculosis in chest radiographs: a survey — Jaeger et al., QIMS", url: "https://qims.amegroups.org/article/view/1813/html" },
  { label: "Optimizing Tuberculosis Detection: A Modified DenseNet121 Approach (Semantic Scholar)", url: "https://www.semanticscholar.org/paper/Optimizing-Tuberculosis-Detection%3A-A-Modified-for-Kumar-Kaur/1764ab0de858681a752b01ab7c7ec228868776e2" },
  { label: "A Novel Method for Detection of Tuberculosis in Chest Radiographs Using Artificial Ecosystem-Based Optimisation of Deep Neural Network Features — Sahlol et al., MDPI Symmetry 2020", url: "https://www.mdpi.com/2073-8994/12/7/1146" },
  { label: "Impact of the ultra-portable digital X-ray with CAD4TB for active case finding for tuberculosis in Nigeria — Frontiers in Digital Health", url: "https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1559203/full" },
  { label: "A Novel Machine Learning Approach for Tuberculosis Segmentation and Prediction Using Chest X-Ray Images — MDPI Applied Sciences", url: "https://www.mdpi.com/2076-3417/11/19/9057" },
  { label: "Enhanced swin transformer based tuberculosis classification with segmentation using chest X-ray — PubMed", url: "https://pubmed.ncbi.nlm.nih.gov/39973770/" },
  { label: "Prompt2SegCXR: Prompt to Segment All Organs and Diseases in Chest X-rays — arXiv", url: "https://arxiv.org/html/2507.00673v1" },
  { label: "Localization and Classification of Abnormalities on Chest X-ray Images Using a Mamba-YOLOvX Model (ResearchGate)", url: "https://www.researchgate.net/publication/391255112_Localization_and_Classification_of_Abnormalities_on_Chest_X-ray_Images_Using_a_Mamba-YOLOvX_Model" },
  { label: "Efficient Tuberculosis Detection Using Chest X-ray Images with Deep Learning Algorithms — University of Ibadan", url: "https://journals.ui.edu.ng/index.php/uijslictr/article/download/1470/1146/4074" },
  { label: "TB-Net: A Tailored, Self-Attention Deep CNN Design for Detection of Tuberculosis Cases From Chest X-Ray Images — Wong et al., Frontiers in AI", url: "https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.827299/full" },
  { label: "Tuberculosis Detection in Chest X-rays Using Vision Transformers with Convolutional Stems and CLAHE-Based Contrast Enhancement (ResearchGate)", url: "https://www.researchgate.net/publication/396761720_Tuberculosis_Detection_in_Chest_X-rays_Using_Vision_Transformers_with_Convolutional_Stems_and_CLAHE-Based_Contrast_Enhancement" },
  { label: "Improved swin transformer-based thorax disease classification with optimal feature selection — PLOS One", url: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327099" },
  { label: "Decision-Aware Vision Mamba with Context-Guided Slot Mixing for Chest X-Ray Screening and Culture-Based Hierarchical Tuberculosis Classification — Jeon et al., MDPI Sensors 2026", url: "https://www.mdpi.com/1424-8220/26/7/2100" },
  { label: "Selective State-Space Models in Medical Image Processing — Preprints.org", url: "https://www.preprints.org/manuscript/202604.0948" },
  { label: "Vision Mamba for efficient Tuberculosis Detection based on Chest X-Rays: A comparative study with CNN and Vision transformers (ResearchGate)", url: "https://www.researchgate.net/publication/394944876_Vision_Mamba_for_efficient_Tuberculosis_Detection_based_on_Chest_X-Rays_A_comparative_study_with_CNN_and_Vision_transformers" },
  { label: "RepViT-CXR: A Channel Replication Strategy for Vision Transformers in Chest X-ray Tuberculosis and Pneumonia Classification — arXiv", url: "https://arxiv.org/html/2509.08234v1" },
  { label: "A Lightweight Hybrid Deep Learning Model for Tuberculosis Detection from Chest X-Rays — Owda et al., MDPI Diagnostics", url: "https://www.mdpi.com/2075-4418/15/24/3216" },
  { label: "From Binary to Multi-Class Classification: A Two-Step Hybrid CNN-ViT Model for Chest Disease Classification (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11639898/" },
  { label: "Use of computer-aided detection software for tuberculosis screening — WHO policy statement", url: "https://www.who.int/publications/i/item/9789240110373" },
  { label: "The Performance of Computer-Aided Detection Digital Chest X-ray Reading Technologies for Triage of Active Tuberculosis — Clinical Infectious Diseases", url: "https://academic.oup.com/cid/article/76/3/e894/6675069" },
  { label: "Beyond accuracy: evaluating the operational feasibility and diagnostic yield of CAD4TB vs. Timika score — Frontiers in Digital Health", url: "https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1748825/full" },
  { label: "Evaluation of chest X-ray with automated interpretation algorithms for mass tuberculosis screening in prisons (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC9904090/" },
  { label: "Comparing different versions of computer-aided detection products when reading chest X-rays for tuberculosis — PLOS Digital Health", url: "https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000067" },
  { label: "Deep Learning Algorithms with Demographic Information Help to Detect Tuberculosis in Chest Radiographs (PMC)", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6352082/" },
  { label: "Revisiting Computer-Aided Tuberculosis Diagnosis — Liu et al., TPAMI 2023 (PubMed)", url: "https://pubmed.ncbi.nlm.nih.gov/37934644/" },
  { label: "Deep Residual Learning for Image Recognition — He et al., CVPR 2016 (arXiv)", url: "https://arxiv.org/abs/1512.03385" },
  { label: "Very Deep Convolutional Networks for Large-Scale Image Recognition — Simonyan & Zisserman, ICLR 2015 (arXiv)", url: "https://arxiv.org/abs/1409.1556" },
  { label: "Densely Connected Convolutional Networks — Huang et al., CVPR 2017 (arXiv)", url: "https://arxiv.org/abs/1608.06993" },
  { label: "U-Net: Convolutional Networks for Biomedical Image Segmentation — Ronneberger et al., 2015 (arXiv)", url: "https://arxiv.org/abs/1505.04597" },
  { label: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale — Dosovitskiy et al., ICLR 2021 (arXiv)", url: "https://arxiv.org/abs/2010.11929" },
  { label: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces — Gu & Dao, 2023 (arXiv)", url: "https://arxiv.org/abs/2312.00752" },
  { label: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks — Tan & Le, ICML 2019 (arXiv)", url: "https://arxiv.org/abs/1905.11946" },
  { label: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows — Liu et al., ICCV 2021 (arXiv)", url: "https://arxiv.org/abs/2103.14030" },
];

const Cite = makeCite(SOURCES);

const FAMILY_TONE: Record<Family, "primary" | "info" | "good" | "warn" | "bad" | "neutral"> = {
  CNN: "primary",
  Segmentation: "info",
  Detection: "good",
  Transformer: "warn",
  SSM: "bad",
  Hybrid: "neutral",
  Commercial: "neutral",
};

function ArchTable() {
  const [family, setFamily] = React.useState<Family | "All">("All");
  const families: (Family | "All")[] = ["All", "CNN", "Segmentation", "Detection", "Transformer", "SSM", "Hybrid", "Commercial"];
  const rows = ARCH_ROWS.filter((r) => family === "All" || r.family === family);

  return (
    <div>
      {/* One filter row above everything it scopes. */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginBottom: "var(--sp-4)" }}>
        {families.map((f) => (
          <button
            key={f}
            onClick={() => setFamily(f)}
            style={{
              all: "unset",
              cursor: "pointer",
              padding: "0.28rem 0.65rem",
              borderRadius: "var(--r-pill)",
              fontFamily: "var(--font-sans)",
              fontSize: "var(--text-xs)",
              fontWeight: "var(--w-semibold)",
              color: family === f ? "var(--primary)" : "var(--ink-3)",
              background: family === f ? "var(--primary-tint)" : "transparent",
              border: `1px solid ${family === f ? "var(--primary-tint-2)" : "var(--line)"}`,
              transition: "background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease)",
            }}
          >
            {f}
          </button>
        ))}
      </div>

      <div style={{ overflowX: "auto", border: "1px solid var(--line)", borderRadius: "var(--r)" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "820px" }}>
          <thead>
            <tr style={{ background: "var(--paper-2)" }}>
              {["Architecture", "Family", "Study", "Dataset", "Reported"].map((h) => (
                <th
                  key={h}
                  style={{
                    textAlign: "left",
                    whiteSpace: "nowrap",
                    padding: "0.6rem 0.8rem",
                    fontFamily: "var(--font-mono)",
                    fontSize: "var(--text-label)",
                    textTransform: "uppercase",
                    letterSpacing: "var(--ls-label)",
                    color: "var(--ink-3)",
                    fontWeight: "var(--w-semibold)",
                    borderBottom: "1px solid var(--line)",
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td style={td(true)}>
                  {r.name}
                  <div style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-xs)", color: "var(--ink-3)", fontWeight: 400, marginTop: "0.15rem" }}>
                    {r.task}
                  </div>
                </td>
                <td style={td()}>
                  <Badge tone={FAMILY_TONE[r.family]}>{r.family}</Badge>
                </td>
                <td style={td()}>{r.study}</td>
                <td style={td()}>
                  {r.dataset}
                  {r.privateData && (
                    <span
                      title="Private or unspecified dataset — this number is not comparable with the others"
                      style={{
                        display: "inline-block",
                        marginLeft: "0.35rem",
                        fontFamily: "var(--font-mono)",
                        fontSize: "0.62rem",
                        color: "var(--warn)",
                        border: "1px solid var(--warn)",
                        borderRadius: "var(--r-sm)",
                        padding: "0 0.25rem",
                        verticalAlign: "middle",
                      }}
                    >
                      private
                    </span>
                  )}
                </td>
                <td style={{ ...td(), fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)" }}>{r.metric}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: "0.6rem", fontFamily: "var(--font-sans)", fontSize: "var(--text-xs)", color: "var(--ink-3)" }}>
        Showing {rows.length} of {ARCH_ROWS.length} architectures.
      </div>
    </div>
  );
}

const td = (strong = false): React.CSSProperties => ({
  verticalAlign: "top",
  padding: "0.6rem 0.8rem",
  fontFamily: "var(--font-sans)",
  fontSize: "var(--text-sm)",
  color: strong ? "var(--ink)" : "var(--ink-2)",
  fontWeight: strong ? "var(--w-semibold)" : "var(--w-regular)",
  lineHeight: "var(--lh-snug)",
  borderTop: "1px solid var(--line)",
});

export function ArchitecturesPage() {
  return (
    <ReviewPage path="/architectures">
      <ReviewHeader
        n={5}
        title="Deep learning architectures for TB detection"
        lead={
          <>
            What each family of network actually does to a chest X-ray, why it was invented, and which one this project
            uses. The diagrams are interactive — hover them; they are trying to show you a mechanism, not decorate the
            page.
          </>
        }
      />

      <ReadSection
        id="cnn"
        n={1}
        title="The convolutional network — where features stopped being hand-written"
        lead="Everything below is a variation on this. If only one figure on this page lands, make it this one."
      >
        <Prose>
          A CNN is handed raw pixels and nothing else. A kernel slides across the image computing dot products, firing
          on local patterns; stack enough of these and the network builds its own hierarchy — edges, then textures,
          then shapes, then whole pathological patterns.<Cite n={2} /> Nobody writes down what a cavity looks like.
          That single property is what ended the feature-engineering era described in the previous section.
        </Prose>

        <Figure
          n="5.1"
          title="Hierarchical feature learning in a CNN"
          caption="Spatial resolution falls left to right while channel depth rises — the network trades knowing exactly where for knowing what. Hover any stage."
        >
          <CNNDiagram />
        </Figure>

        <Prose>
          <strong>AlexNet</strong> established the template — stacked convolutions, max-pooling, ReLU, dense layers —
          using large 11×11 receptive fields early on.<Cite n={1} /> <strong>GoogLeNet</strong> added the Inception
          module, running 1×1, 3×3 and 5×5 convolutions in parallel and concatenating them, letting the network pick
          its own receptive-field size — useful when TB ranges from tiny calcifications to lobe-filling
          consolidation.<Cite n={1} /> Their ensemble is the Lakhani & Sundaram result from the previous section.
        </Prose>
        <Prose style={{ marginBottom: 0 }}>
          <strong>VGG</strong> then made a structural argument: throw away large kernels and stack only 3×3 filters at
          stride 1.<Cite n={32} /> Two stacked 3×3 layers see the same region as one 5×5; three see the same as a 7×7 —
          but with a ReLU between each, so the decision function gets more non-linear <em>and</em> uses fewer
          parameters than the single wide layer.<Cite n={32} /> It is old and it remains startlingly competitive: in a
          2025 benchmark over 4,200 CXRs, VGG-16 beat newer models at 99.4% accuracy.<Cite n={4} /> Its cost is memory —
          which rules it out for the point-of-care hardware discussed below.<Cite n={3} />
        </Prose>
      </ReadSection>

      <ReadSection
        id="resnet"
        n={2}
        title="ResNet — the wire that made depth possible"
        lead="One connection, and networks went from ~20 layers to 150+. This project's detector backbone is built on it."
      >
        <Prose>
          Stacking layers should help, and past a point it didn't — deeper plain networks trained <em>worse</em>. The
          cause is the vanishing gradient: on the backward pass, gradients are multiplied through every layer, and
          repeated multiplication by small numbers drives them toward zero, so early layers stop learning.<Cite n={31} />
        </Prose>
        <Prose>
          He et al.'s fix is almost embarrassingly small: add a wire that carries the input around the convolutions and
          adds it back.<Cite n={31} /> Instead of forcing the layers to produce the whole mapping H(x), let them
          produce the <em>residual</em> F(x) = H(x) − x, and recover H(x) = F(x) + x.
        </Prose>

        <Figure
          n="5.2"
          title="The residual block — hover the skip connection"
          caption={
            <>
              The animation on the skip is the point: on the backward pass the gradient reaches earlier layers{" "}
              <em>without</em> being multiplied through the convolutions. Adding depth can no longer make training
              error worse, because a useless block can drive F(x) to zero and pass x straight through.
            </>
          }
        >
          <ResNetDiagram />
        </Figure>

        <Prose>
          ResNet-50 and ResNet-152 are the workhorses of modern TB research. On multi-class problems — separating
          normal, viral pneumonia, bacterial pneumonia and TB — ResNets routinely exceed 96% accuracy.<Cite n={5} /> In
          semi-supervised training over 100,000+ unlabelled radiographs, the ResNet-backed <strong>TBNet</strong>{" "}
          reached AUC 0.91 on an external test set<Cite n={5} /> — a lower number than the private-dataset results
          nearby, and a much more trustworthy one, because it is external.
        </Prose>

        <Prose>
          <strong>DenseNet</strong> pushes the same instinct further: connect every layer to every subsequent layer, so
          a network of L layers has L(L+1)/2 connections instead of L.<Cite n={33} /> Features computed early — the
          sharp boundary of a calcified granuloma — arrive at the classifier undegraded. A small growth rate and 1×1
          bottleneck layers keep it affordable, and the aggressive feature reuse makes it notably resistant to
          overfitting on small datasets.<Cite n={7} /> DenseNet-169 with an SVM head is the source of the 99.91%
          figure in the table below.<Cite n={7} />
        </Prose>

        <Callout kind="note" title="Why this section is here">
          YOLOv8's backbone is residual. Every gain this project has measured runs through that wire — so the skip
          connection is not background reading, it is a component of the thing on the Results page.
        </Callout>
      </ReadSection>

      <ReadSection
        id="unet"
        n={3}
        title="U-Net — answering for every pixel"
        lead="Not a classifier. A segmenter: one label per pixel, not one per image."
      >
        <Prose>
          Classification says "TB, 0.94". Segmentation draws the boundary. U-Net's shape follows from that demand: a
          contracting encoder builds semantic understanding while destroying spatial precision, and an expanding
          decoder upsamples back to full resolution — but the decoder alone cannot recover detail that pooling threw
          away.<Cite n={34} /> The fix is skip connections again, this time copying high-resolution feature maps
          laterally from encoder to decoder.<Cite n={34} />
        </Prose>

        <Figure
          n="5.3"
          title="U-Net — hover the skip ladder"
          caption="Deep semantics from the bottleneck, sharp edges from the skips. The same 'carry information around the lossy part' trick as ResNet, used for a different job."
        >
          <UNetDiagram />
        </Figure>

        <Prose style={{ marginBottom: 0 }}>
          In TB pipelines U-Net is usually not the diagnostic model — it is the <em>preprocessing</em> step. Pacemakers,
          external hardware, the cardiac silhouette and rib margins all confound classifiers into false positives, so
          U-Net masks and crops the lung parenchyma first, and only the lungs go downstream.<Cite n={3} /> On Montgomery
          and Shenzhen, modern implementations reach 96.35% accuracy, Jaccard 90.38%, Dice 94.88%.<Cite n={10} /> Put an
          Attention U-Net in front of a Swin classifier and reported diagnostic accuracy exceeds 99%.<Cite n={11} />
        </Prose>

        <Callout kind="note" title="A parked idea for this project" style={{ marginTop: "var(--sp-5)" }}>
          Lung-segmentation masking is on our list of data-centric levers precisely because of the failure mode in the
          Results log: the detector fires on sick-but-not-TB lungs 2–3× more than on healthy ones. Masking away
          everything that is not lung is one way to attack that.
        </Callout>
      </ReadSection>

      <ReadSection
        id="detectors"
        n={4}
        title="Detectors — where this project lives"
        lead="Boxes, not scores. The difference between 'this X-ray has TB' and 'the TB is here'."
      >
        <Prose>
          A classifier gives one number for the whole image. A detector outputs coordinates: a box drawn over the
          cavity. That matters for more than tidiness — it is what makes the output checkable. A radiologist can
          immediately verify a box; they cannot verify a 0.94.<Cite n={13} /> Localisation is explainability you get
          for free.
        </Prose>

        <Figure
          n="5.4"
          title="Two-stage vs one-stage detection"
          caption="Switch between the tabs — the two pipelines are the same diagram with a different middle. The presence or absence of a proposal step is the entire architectural difference."
        >
          <DetectorDiagram />
        </Figure>

        <CardGrid cols={2}>
          <InfoCard title="Faster R-CNN — propose, then classify">
            <Bullets
              items={[
                "A Region Proposal Network scans the feature map for anchor boxes that might contain something.",
                "A second head then classifies each proposal and refines its coordinates.",
                <>
                  Precise, classically strong on small lesions, and computationally heavy. Xie et al. reached AUC 0.977
                  and 92.6% accuracy on Shenzhen + Montgomery.<Cite n={3} />
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard tone="primary" title="YOLO — one pass, and this project's choice">
            <Bullets
              items={[
                "Detection framed as a single regression: box coordinates and class probabilities straight from the image grid.",
                <>
                  Fast enough for real time; general implementations report ~68% mAP, trading precision for
                  speed.<Cite n={13} />
                </>,
                <>
                  Variants like <strong>YOLO-CXR</strong> add Selective Feature Fusion specifically to catch small,
                  early lesions.<Cite n={13} />
                </>,
              ]}
            />
          </InfoCard>
        </CardGrid>

        <KeyIdea>
          Our own detector is a YOLOv8n. It reaches Active mAP50 = 0.745 on a sealed test set — nowhere near the 99%
          accuracies in the table below, and not comparable to them either: those are binary image-level
          classification, this is lesion localisation on 799 training images.
        </KeyIdea>
      </ReadSection>

      <ReadSection
        id="transformers"
        n={5}
        title="Transformers — TB is a whole-lung disease"
        lead="A CNN filter sees its own neighbourhood. Attention sees everything at once, and that turns out to matter clinically."
      >
        <Prose>
          A Vision Transformer cuts the radiograph into patches, treats them like words in a sentence, and uses
          self-attention to compute how strongly every patch relates to every other patch — regardless of how far
          apart they are.<Cite n={35} /> Positional embeddings restore the spatial awareness that flattening
          destroyed.
        </Prose>
        <Prose>
          This is not an abstract advantage for TB. The disease frequently presents as diffuse bilateral interstitial
          patterns or miliary mottle spanning the whole thorax. A CNN, restricted to local receptive fields in its
          early layers, must go deep before it can relate an upper-lobe consolidation to a distant lower-lobe
          effusion. A ViT does it in one step.<Cite n={5} />
        </Prose>

        <Figure
          n="5.5"
          title="Self-attention — hover a patch to see what it looks at"
          caption={
            <>
              Hover one of the outlined lesion patches: it attends strongly to the <em>other</em> lesion in the
              opposite lung despite the distance. Hover healthy tissue and attention stays local and weak. Attention is
              learnt, not fixed.
            </>
          }
        >
          <AttentionDiagram />
        </Figure>

        <Prose>
          The mechanism is multi-head self-attention: patches are projected to queries, keys and values, and each
          output is a weighted sum of values where the weight comes from query–key compatibility, softmax(QKᵀ/√d
          <sub>k</sub>)V.<Cite n={35} /> The catch is in the figure: every patch against every patch is quadratic in
          patch count. <strong>Swin</strong> answers this by computing attention inside local windows that shift
          between layers, restoring a CNN-like locality bias and making high-resolution medical images
          affordable.<Cite n={38} />
        </Prose>
        <Prose style={{ marginBottom: 0 }}>
          Reported results are extraordinary and mostly unverifiable: ViT with CLAHE preprocessing at 99.90% average
          precision and 99.84% accuracy<Cite n={16} />; EnSTrans at 99.05%<Cite n={11} />; TB-Net at 99.86% accuracy
          and 100% sensitivity.<Cite n={15} /> ViTs lack the CNN's built-in inductive biases, so they need large-scale
          pretraining to work at all — which is exactly why hybrids exist.<Cite n={32} />
        </Prose>
      </ReadSection>

      <ReadSection
        id="mamba"
        n={6}
        title="Mamba — global context without the quadratic bill"
        lead="The newest family here, and the one with the most interesting result for this project specifically."
      >
        <Prose>
          Mamba is a selective state space model. It keeps the ability to relate distant parts of an image but does it
          in <strong>linear</strong> time, by treating the input as a sequence flowing through a hidden state governed
          by continuous-time dynamics, discretised into a linear recurrence.<Cite n={36} /> The "selective" part is the
          innovation: the state matrices are input-dependent, so the model dynamically decides what to keep and what
          to discard — filtering out healthy tissue and scanner artefacts while holding pathology in
          memory.<Cite n={19} /> For 2D images, Vision Mamba scans the grid in multiple directions so no spatial
          context is lost, and a hardware-aware algorithm manages GPU SRAM/HBM traffic directly.<Cite n={19} />
        </Prose>

        <Figure
          n="5.6"
          title="Why linear beats quadratic — drag the slider"
          caption={
            <>
              At short sequences the two are nearly identical and attention's global view is almost free. Grow the
              sequence — which is what happens when you stop downsampling a chest X-ray — and they separate hard. This
              is the whole argument for SSMs in medical imaging, and it is why Mamba reports ~80% less GPU memory than
              an equivalent ViT at comparable accuracy.<Cite n={20} />
            </>
          }
        >
          <MambaDiagram />
        </Figure>

        <Callout kind="decision" title="The active-vs-inactive result — the one worth paying attention to">
          Telling <strong>active</strong> TB from the fibrotic scarring of a <strong>cured</strong> infection is the
          hardest call in the field, and traditional CNNs fail it: at a local textural level the tissue deformation
          looks the same, so they emit false positives on people who were cured years ago.<Cite n={18} /> Jeon et al.'s
          Vision Mamba with Context-Guided Slot Mixing attacks it by separating subtle pathological features from
          global anatomical context, reporting <strong>97.04% specificity</strong> and a Youden index of 79.55% on
          active vs inactive — beating ResNet-152 and ViT-B baselines — alongside 92.96% multi-class
          accuracy.<Cite n={18} /> The stakes are unnecessary quarantines and expensive molecular tests for people with
          a TB history.
        </Callout>

        <Prose style={{ marginBottom: 0 }}>
          This maps directly onto our own problem. TBX11K's two classes are exactly <code>ActiveTuberculosis</code> and{" "}
          <code>ObsoletePulmonaryTuberculosis</code>, and our detector's Obsolete AP sits around 0.20 — effectively
          noise. The literature says this distinction is genuinely hard and that global context is what cracks it.
        </Prose>
      </ReadSection>

      <ReadSection
        id="hybrids"
        n={7}
        title="Hybrids and the edge"
        lead="Two pressures the benchmark tables never show: not enough data, and not enough hardware."
      >
        <CardGrid cols={2}>
          <InfoCard title="CNN–ViT hybrids — the data-efficiency fix">
            <Bullets
              items={[
                "A CNN stem extracts local features (edges, calcifications, cavity texture); Transformer blocks then reason globally over them.",
                <>
                  This injects convolutional inductive biases — translation invariance, locality — into the
                  Transformer, so it needs far less data to train.<Cite n={23} />
                </>,
                <>
                  <strong>RepViT-CXR</strong> claims new SOTA on TB and pneumonia by beating exactly this data-hunger
                  problem.<Cite n={21} />
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard title="Lightweight nets — the hardware fix">
            <Bullets
              items={[
                <>
                  <strong>MobileNet</strong> factorises convolution into depthwise + pointwise steps, collapsing cost
                  and size.<Cite n={3} />
                </>,
                <>
                  <strong>EfficientNet</strong> scales depth, width and resolution together with fixed coefficients
                  rather than one at a time.<Cite n={37} />
                </>,
                <>
                  <strong>GhostNet + MobileViT</strong> reports {">"}99% accuracy at 7.73M parameters and 282.11M
                  FLOPs.<Cite n={22} /> Sahlol et al. distilled 50,000 MobileNet features to under 25 and still hit
                  94.1% on external validation.<Cite n={8} />
                </>,
              ]}
            />
          </InfoCard>
        </CardGrid>

        <KeyIdea>
          The regions with the highest TB burden frequently have no reliable internet for cloud inference. A
          lightweight net running offline on a battery-powered ultra-portable X-ray unit in a remote village is
          arguably worth more than a marginal AUC gain in a model that needs a datacentre.
        </KeyIdea>
      </ReadSection>

      <ReadSection
        id="table"
        n={8}
        title="Every architecture, as reported"
        lead="Filter by family. Read the caveat under the table before you read the numbers."
      >
        <Callout kind="landmine" title="These numbers are NOT comparable with each other">
          They come from different studies on different datasets of wildly different size and difficulty. Rows marked{" "}
          <em>private</em> report on private or unspecified data, so nobody can reproduce or verify them — including
          the headline 99.4% (VGG-16), 99.91% (DenseNet-169) and ~99.9% (ViT hybrids). Most are binary image-level
          classification; a few are localisation, which is a strictly harder task. Sorting this table would imply a
          ranking that the underlying evidence does not support, so it does not sort.
        </Callout>

        <div style={{ marginTop: "var(--sp-5)" }}>
          <ArchTable />
        </div>

        <div style={{ marginTop: "var(--sp-6)" }}>
          <Collapsible title="The foundational public datasets" subtitle="What each architecture above was actually trained on">
            <DataTable
              head={["Dataset", "Size", "Composition", "Task", "Primary role"]}
              rows={FOUNDATION_DATASETS.map((d) => [d.name, d.size, d.composition, d.tasks, d.role])}
            />
          </Collapsible>
        </div>

        <Figure
          n="5.7"
          title="The gap between a benchmark number and a verified one"
          caption={
            <>
              Every bar is a headline accuracy from the table above. The teal bars were measured on{" "}
              <strong>named public data or external validation</strong>; the grey bars come from private or
              unspecified datasets. The grey ones are systematically higher — which is a fact about evidence quality,
              not about the models.
            </>
          }
          source="Values as reported by each source study; see the table above for the full metrics."
          table={{
            head: ["Architecture", "Reported", "Data"],
            rows: ARCH_ROWS.filter((r) => r.value !== undefined).map((r) => [
              r.name,
              r.metric,
              r.privateData ? "private / unspecified" : "public / external",
            ]),
          }}
        >
          <BarChart
            max={100}
            unit="%"
            labelWidth={190}
            bars={ARCH_ROWS.filter((r) => r.value !== undefined).map((r) => ({
              label: r.name,
              value: r.value!,
              display: `${r.value}`,
              emphasis: !r.privateData,
              note: r.dataset,
            }))}
          />
        </Figure>
      </ReadSection>

      <ReadSection
        id="clinical"
        n={9}
        title="What actually reached the clinic"
        lead="The WHO does not care about your mAP. It cares about sensitivity at a fixed specificity, at a price."
      >
        <Prose>
          In 2021, updated in 2025, the WHO formally recommended CAD as an alternative to a human reader for TB
          screening and triage in people aged 15 and over.<Cite n={24} /> The bar is the Target Product Profile:{" "}
          <strong>at least 90% sensitivity and 70% specificity</strong>.<Cite n={25} /> That is the number a deployed
          system is judged on — not accuracy, which is meaningless when almost everyone screened is healthy.
        </Prose>

        <CardGrid cols={2}>
          <InfoCard title="CAD4TB (Delft Imaging)">
            <Bullets
              items={[
                <>
                  Began as classical ML — segmentation, texture, k-NN/SVM, a 0–100 abnormality score.<Cite n={28} />
                </>,
                <>
                  Rebuilt on CNNs from v6 (2018). AUC rose from <strong>0.823 (v6) to 0.903 (v7)</strong>, matching or
                  exceeding expert radiologists at triage.<Cite n={25} />
                </>,
                <>
                  Economics, in Pakistan: 132 subjects/day at <strong>$5.95 per screen</strong> at the 90%-sensitivity
                  threshold.<Cite n={25} />
                </>,
              ]}
            />
          </InfoCard>
          <InfoCard title="qXR (Qure.ai) & Lunit INSIGHT CXR">
            <Bullets
              items={[
                "Built deep from the start; multi-label, so they also localise discrete findings — calcifications, cavities, pleural effusions.",
                <>
                  In large screening cohorts (e.g. incarcerated populations in Brazil) they exceed{" "}
                  <strong>70% specificity at a fixed 90% sensitivity</strong> — clearing the WHO TPP.<Cite n={27} />
                </>,
                "The payoff is economic: safely screening out negatives means far fewer costly GeneXpert confirmations.",
              ]}
            />
          </InfoCard>
        </CardGrid>

        <Callout kind="note" title="Explainability is not a nice-to-have" style={{ marginTop: "var(--sp-5)" }}>
          Grad-CAM alongside ResNet, ViT and Mamba CGSM is now standard practice.<Cite n={18} /> Confirming that a
          network attended to an upper-lobe cavitation rather than a clavicle artefact is what earns radiologist
          trust — and it is the same argument for detection over classification that this project is built on: a box
          is checkable.
        </Callout>
      </ReadSection>

      <ReadSection
        id="takeaways"
        n={10}
        title="Where this leaves our own work"
        lead="Five things this survey changes about how we read our own numbers."
      >
        <CardGrid cols={2}>
          <InfoCard tone="primary" title="Local → global is the through-line">
            <Prose style={{ margin: 0, fontSize: "var(--text-sm)" }}>
              CNN → ViT → Mamba is one argument repeated: TB is a whole-lung disease, and models that can relate
              distant regions do better on it. Our YOLOv8n is firmly on the local end of that line.<Cite n={19} />
            </Prose>
          </InfoCard>
          <InfoCard tone="primary" title="Our Obsolete class is a known-hard problem">
            <Prose style={{ margin: 0, fontSize: "var(--text-sm)" }}>
              Active vs inactive defeats CNNs in the literature too. Our ~0.20 Obsolete AP is consistent with that —
              it is the field's open problem, not obviously our bug.<Cite n={18} />
            </Prose>
          </InfoCard>
          <InfoCard tone="primary" title="Segmentation-first is a real lever">
            <Prose style={{ margin: 0, fontSize: "var(--text-sm)" }}>
              U-Net masking is standard practice precisely to stop models firing on non-lung structure — our
              highest-value parked idea, aimed straight at the sick-lung false alarms.<Cite n={3} />
            </Prose>
          </InfoCard>
          <InfoCard tone="primary" title="Two-stage is the field's answer too">
            <Prose style={{ margin: 0, fontSize: "var(--text-sm)" }}>
              Commercial systems screen with a scorer and localise separately. Our classifier → detector plan is the
              same shape, arrived at independently from our own exp2 result.<Cite n={27} />
            </Prose>
          </InfoCard>
        </CardGrid>

        <Callout kind="landmine" title="And the one that matters most" style={{ marginTop: "var(--sp-5)" }}>
          A field where the published numbers cluster at 99%+ while radiologists score 68.7% on an open
          benchmark<Cite n={30} /> is a field with a measurement problem, not a solved one. That is why this project
          decides on a sealed test set, reports multi-seed bands rather than best runs, and keeps a public list of its
          own retractions. Our 0.745 is a smaller number than almost everything on this page — and we can tell you
          exactly what it means.
        </Callout>
      </ReadSection>

      <Bibliography sources={SOURCES} />
    </ReviewPage>
  );
}
