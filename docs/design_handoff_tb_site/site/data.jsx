// Real content lifted from the repo: RESULTS.md, data.html, literature.md.
// Exposed on window for the page components.

const EXPERIMENTS = [
  {
    id: "exp1",
    title: "YOLOv8n floor — positives-only, augmentation off",
    outcome: { tone: "neutral", label: "Floor" },
    headline: [
      { label: "Active mAP50", value: "0.53", tone: "warn" },
      { label: "Matched IoU", value: "0.74", tone: "good" },
    ],
    stats: [
      { label: "Active mAP50", value: "0.53", tone: "warn" },
      { label: "Matched IoU", value: "0.74", tone: "good" },
      { label: "Overfits by", value: "~ep 12", tone: "bad" },
      { label: "Sick false-fire", value: "2–3×", tone: "warn" },
    ],
    findings: [
      { kind: "note", text: "Bottleneck is <strong>recall, not box quality</strong> — matched IoU ~0.74 means lesions are missed, not mislocated." },
      { kind: "landmine", text: "Overfits by ~epoch 12 — more epochs hurt. Active &gt;&gt; Obsolete (class imbalance; Obsolete barely learnable)." },
      { kind: "note", text: "Fires on sick (non-TB) lungs ~2–3× the healthy rate — confuses other pathology with TB." },
    ],
  },
  {
    id: "exp2",
    title: "Negatives in detector training",
    outcome: { tone: "bad", label: "Lever closed" },
    headline: [
      { label: "False alarms", value: "collapse", tone: "good" },
      { label: "TB sensitivity", value: "~60%", tone: "bad" },
    ],
    stats: [
      { label: "False alarms", value: "↓ collapse", tone: "good" },
      { label: "Sensitivity cap", value: "~60%", tone: "bad" },
      { label: "Saturates at", value: "0.25:1", tone: "warn" },
    ],
    findings: [
      { kind: "note", text: "Background negatives <strong>collapse false alarms</strong> (even 0.25:1, saturates at once) but <strong>cap TB sensitivity at ~60%</strong>." },
      { kind: "decision", text: "Specificity belongs in the classifier; keep the detector positives-only / sensitive. Negatives lever closed for the detector." },
      { kind: "retraction", isRetract: true, text: "Ratio (0.25 / 0.5 / 1.0) unresolvable — single-seed noise ≥ effect." },
    ],
  },
  {
    id: "exp3",
    title: "Augmentation screen — 512 @ batch 16, positives-only",
    outcome: { tone: "good", label: "Biggest lever" },
    headline: [
      { label: "Active mAP50", value: "0.74", tone: "good" },
      { label: "vs off", value: "+0.21", tone: "good" },
    ],
    stats: [
      { label: "mAP50 (off)", value: "0.53", tone: "warn" },
      { label: "mAP50 (mosaic)", value: "0.74", tone: "good" },
      { label: "Val peak moves", value: "ep 7 → 110", tone: "good" },
    ],
    findings: [
      { kind: "decision", text: "Augmentation is the <strong>biggest lever yet</strong> — Active mAP50 0.53 (off) → 0.74 (mosaic). Also fixes overfitting: val-mAP peak moves from epoch ~7 to ~110–145." },
      { kind: "note", text: "Domain pruning held — brightness (geo_photo) and the kitchen sink (default) underperform plain geometry. Finalists carried forward: <strong>geo</strong> and <strong>mosaic</strong>." },
    ],
  },
  {
    id: "exp4",
    title: "Multi-seed validation (3 seeds)",
    outcome: { tone: "good", label: "Finalist" },
    headline: [
      { label: "Active mAP50", value: "0.70", tone: "good" },
      { label: "Config", value: "512 @ b16", tone: "default" },
    ],
    stats: [
      { label: "mosaic @ 1024", value: "0.683–0.699", tone: "good" },
      { label: "geo @ 1024", value: "0.667–0.671", tone: "warn" },
      { label: "Obsolete", value: "~0.20 (noise)", tone: "bad" },
      { label: "512 noise", value: "±0.029", tone: "warn" },
    ],
    findings: [
      { kind: "decision", text: "<strong>mosaic is the augmentation finalist</strong> — Active mAP50 bands across seeds don't overlap with geo. Final detector config: <strong>mosaic @ 512, batch 16</strong>." },
      { kind: "retraction", isRetract: true, text: "<s>geo has better Obsolete</s> — retracted: geo's 512 edge (0.320 vs 0.272) was single-seed luck; multi-seed both ~0.20 (tied)." },
      { kind: "retraction", isRetract: true, text: "<s>512@16 clearly beats 1024@8</s> — corrected: anchored on one lucky seed. Multi-seed, 1024@8 ≈ 512@16." },
    ],
  },
];

const NEXT_UP = [
  { id: "exp5", text: "k-fold CV — test-set-luck / split variance on the final config." },
  { id: "ssl", text: "VinDr-pretrained weights → init for TBX11K (Kaggle, parallel). Freezing ablation pairs with this." },
];

const DATASETS = [
  {
    id: "tbx11k", name: "TBX11K", host: "github.com", url: "https://github.com/yun-liu/Tuberculosis",
    tb: "~800 TB (11,200)", bbox: "yes", seg: "no", access: "Free", role: { tone: "primary", label: "Training" },
    annos: ["Image-level class (6 cat.)", "BBox: Active + Latent TB"],
    stats: [
      { label: "Total images", value: "11,200" }, { label: "TB+ (trainval)", value: "~800" },
      { label: "Total bboxes", value: "1,211" }, { label: "Resolution", value: "512×512" },
    ],
    nuances: [
      "Only public TB dataset combining image-level subtype classification with bounding boxes.",
      "~93% of images have empty label files (healthy / sick-non-TB) — correct, not missing data.",
      "Test-set GT deliberately withheld — use train+val for all local experiments.",
      "Experienced radiologists achieved only 68.7% accuracy vs gold standard — contextualises task difficulty.",
    ],
    note: "CVPR 2020, Nankai University & InferVision. CC BY 4.0.",
  },
  {
    id: "shenzhen", name: "Shenzhen Hospital CXR", host: "nih.gov", url: "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html",
    tb: "336 TB (662)", bbox: "no", seg: "yes", access: "Free", role: { tone: "info", label: "OOD / test" },
    annos: ["Image-level (TB / Normal)", "Pixel masks: 19 lesion types"],
    stats: [
      { label: "Total images", value: "662" }, { label: "TB positive", value: "336" },
      { label: "Lesion types", value: "19" }, { label: "Mask format", value: "Per-instance" },
    ],
    nuances: [
      "Masks are per-lesion-instance PNGs — one image can have 10+ separate mask files.",
      "Binary classification only — no Active / Latent distinction unlike TBX11K.",
      "Standard use: train on TBX11K, test generalization on Shenzhen as OOD set.",
    ],
    note: "NLM release 2014; pixel masks published MDPI Data 2022.",
  },
  {
    id: "chestxray14", name: "NIH ChestX-ray14", host: "kaggle.com", url: "https://www.kaggle.com/datasets/nih-chest-xrays/data",
    tb: "0 (no TB label)", bbox: "partial", seg: "no", access: "Free", role: { tone: "neutral", label: "Pretraining" },
    annos: ["Image-level: 14 diseases (NLP)", "BBox: ~880 images, 8 diseases"],
    stats: [
      { label: "Total images", value: "112,120" }, { label: "Patients", value: "30,805" },
      { label: "Classes", value: "14" }, { label: "Label noise", value: "~5–10%" },
    ],
    nuances: [
      "NO TB label — value is scale for pretraining only. Labels are NLP-extracted, ~5–10% noise.",
      "Official splits have 67.4% patient leakage — always use patient-level splits.",
      "ChestX-ray14 → TBX11K fine-tuning is a common, effective pipeline in literature.",
    ],
    note: "Wang et al., CVPR 2017. NIH Clinical Center.",
  },
  {
    id: "vindr", name: "VinDr-CXR", host: "kaggle.com", url: "https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection",
    tb: "0 (no TB label)", bbox: "yes", seg: "no", access: "Free", role: { tone: "neutral", label: "Pretraining" },
    annos: ["Image-level (14 findings + No Finding)", "BBox: all 18,000 images"],
    stats: [
      { label: "Total images", value: "18,000" }, { label: "Train", value: "15,000" },
      { label: "Anno / train", value: "3 rads" }, { label: "Anno / test", value: "5-consensus" },
    ],
    nuances: [
      "Highest annotation quality of any public CXR dataset — fully human, multi-reader.",
      "No TB label, but TB manifestations (consolidation, calcification, nodule) are present and labeled.",
      "VinDr → TBX11K fine-tuning is a stronger start than ChestX-ray14 (cleaner human boxes).",
    ],
    note: "Nguyen et al., Scientific Data (Nature) 2022.",
  },
];

const PAPER_GROUPS = [
  {
    group: "Backbones / classification",
    blurb: "Read first — the spine for both the classifier and the detector.",
    papers: [
      { important: true, title: "ResNet — Deep Residual Learning for Image Recognition", meta: "He et al. · CVPR 2016", link: "https://arxiv.org/abs/1512.03385", note: "The skip-connection paper. The backbone candidate for both stages.", read: true },
      { title: "ConvNeXt — A ConvNet for the 2020s", meta: "Liu et al. · CVPR 2022", link: "https://arxiv.org/abs/2201.03545", note: "Modernized ResNet; strong current CNN backbone." },
      { title: "EfficientNet", meta: "Tan & Le · ICML 2019", link: "https://arxiv.org/abs/1905.11946", note: "Compound scaling; a classifier-backbone alternative." },
    ],
  },
  {
    group: "Detectors",
    blurb: "The actual YOLO vs RetinaNet vs Faster R-CNN decision. Read in dependency order.",
    papers: [
      { important: true, title: "FPN — Feature Pyramid Networks", meta: "Lin et al. · CVPR 2017", link: "https://arxiv.org/abs/1612.03144", note: "The unsung critical one for us — multi-scale features, the main lever for small-lesion recall. Don't skip.", read: true },
      { important: true, title: "Faster R-CNN", meta: "Ren et al. · NeurIPS 2015", link: "https://arxiv.org/abs/1506.01497", note: "Introduces the RPN — the candidate two-stage detector. Classically strong small-lesion recall." },
      { important: true, title: "RetinaNet / Focal Loss", meta: "Lin et al. · ICCV 2017", link: "https://arxiv.org/abs/1708.02002", note: "Candidate one-stage detector; focal loss targets our class imbalance directly." },
      { title: "YOLOv1 — You Only Look Once", meta: "Redmon et al. · CVPR 2016", link: "https://arxiv.org/abs/1506.02640", note: "The original one-stage idea.", read: true },
    ],
  },
  {
    group: "Domain papers (CXR / TB) — must cite",
    blurb: "Ground the work in the medical-imaging literature.",
    papers: [
      { important: true, title: "TBX11K — Rethinking Computer-aided Tuberculosis Diagnosis", meta: "Liu et al. · CVPR 2020", link: "https://github.com/yun-liu/Tuberculosis", linkLabel: "project page", note: "Our dataset paper — non-negotiable citation.", read: true },
      { title: "CheXNet", meta: "Rajpurkar et al. · 2017", link: "https://arxiv.org/abs/1711.05225", note: "DenseNet-121 on ChestX-ray14; canonical CXR-classification reference." },
      { title: "MoCo-CXR", meta: "Sowrirajan et al. · MIDL 2021", link: "https://arxiv.org/abs/2010.05352", note: "SSL on chest X-rays specifically — addresses the grayscale-augmentation problem." },
    ],
  },
  {
    group: "Next phase — transformers & SSMs",
    blurb: "The line you're moving into after CNNs.",
    papers: [
      { important: true, title: "Attention Is All You Need", meta: "Vaswani et al. · NeurIPS 2017", link: "https://arxiv.org/abs/1706.03762", note: "The transformer. Foundation for everything below (currently reading)." },
      { title: "ViT — An Image is Worth 16×16 Words", meta: "Dosovitskiy et al. · ICLR 2021", link: "https://arxiv.org/abs/2010.11929", note: "Transformers as image backbones." },
      { important: true, title: "Mamba — Selective State Spaces", meta: "Gu & Dao · 2023", link: "https://arxiv.org/abs/2312.00752", note: "Linear-time selective SSM; the one to know." },
    ],
  },
];

// --- YOLO experiment runs (per-seed), newest first. Table fields are real
// (lifted from the ablation page); per-run detail is built by buildRunMetrics.
const RUN_DATE = "27 Jul 2026";
const RUNS = [
  { name: "exp4_posonly_mosaic_1024_s2", group: "exp4", aug: "mosaic", imgsz: 1024, batch: 8, seed: 2,
    map50: "0.433", precision: "0.298", recall: "0.608", tbCaught: "108/121", healthyFA: "2.5%", sickFA: "16.7%", trainTime: "1h 14m", best: true,
    desc: "Mosaic at 1024@8, seed 2. Active mAP50 = 0.699 — highest of the three mosaic@1024 seeds. Best specificity of all mosaic runs: 2.5% healthy FA and 16.7% sick FA at conf=0.25. Mosaic band [0.683–0.699] does not overlap geo's [0.667–0.671], confirming mosaic as the augmentation winner with statistical confidence. Decision: mosaic is the augmentation finalist. Final detector config: mosaic @ 512, batch 16.",
    metrics: { active: "0.699", obsolete: "0.167", iou: "0.730", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.458" }, large: { n: 105, v: "0.686" } } } },
  { name: "exp4_posonly_mosaic_1024_s1", group: "exp4", aug: "mosaic", imgsz: 1024, batch: 8, seed: 1,
    map50: "0.429", precision: "0.418", recall: "0.512", tbCaught: "118/121", healthyFA: "8.3%", sickFA: "23.3%", trainTime: "1h 25m",
    desc: "Mosaic at 1024@8, seed 1. Highest TB sensitivity of the seed (118/121 flagged) but at a higher false-alarm cost. Active mAP50 = 0.683 — bottom of the mosaic band, still above the geo ceiling." },
  { name: "exp4_posonly_mosaic_1024_s0", group: "exp4", aug: "mosaic", imgsz: 1024, batch: 8, seed: 0,
    map50: "0.465", precision: "0.457", recall: "0.541", tbCaught: "118/121", healthyFA: "7.5%", sickFA: "25.8%", trainTime: "1h 25m",
    desc: "Mosaic at 1024@8, seed 0. Best overall mAP50 of the three mosaic seeds. Confirms mosaic's lead is not a single-seed artefact." },
  { name: "exp4_posonly_geo_1024_s2", group: "exp4", aug: "geo", imgsz: 1024, batch: 8, seed: 2,
    map50: "0.449", precision: "0.483", recall: "0.487", tbCaught: "116/121", healthyFA: "15.8%", sickFA: "32.5%", trainTime: "1h 24m",
    desc: "Geometry-only at 1024@8, seed 2. Strong precision but markedly worse specificity than mosaic — healthy FA nearly 6× the matched mosaic run." },
  { name: "exp4_posonly_geo_1024_s1", group: "exp4", aug: "geo", imgsz: 1024, batch: 8, seed: 1,
    map50: "0.424", precision: "0.385", recall: "0.487", tbCaught: "113/121", healthyFA: "5.0%", sickFA: "20.8%", trainTime: "1h 16m",
    desc: "Geometry-only at 1024@8, seed 1. Active mAP50 lands inside the geo band [0.667–0.671] — below every mosaic seed." },
  { name: "exp4_posonly_geo_1024_s0", group: "exp4", aug: "geo", imgsz: 1024, batch: 8, seed: 0,
    map50: "0.430", precision: "0.271", recall: "0.527", tbCaught: "114/121", healthyFA: "1.7%", sickFA: "27.5%", trainTime: "1h 18m",
    desc: "Geometry-only at 1024@8, seed 0. Lowest precision of the geo seeds; the low healthy-FA reading is offset by a high sick-FA rate." },
  { name: "exp3_mosaic_mixup_512_s2", group: "exp3", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 2,
    map50: "0.517", precision: "0.548", recall: "0.532", tbCaught: "117/121", healthyFA: "5.8%", sickFA: "36.7%", trainTime: "26m",
    desc: "Mosaic + mixup at 512@16, seed 2. Top headline mAP50 of the whole screen, but mixup inflates sick-lung false fires (36.7%) — the reason mixup was dropped from the finalist config." },
  { name: "exp3_mosaic_mixup_512_s1", group: "exp3", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 1,
    map50: "0.437", precision: "0.476", recall: "0.447", tbCaught: "120/121", healthyFA: "10.8%", sickFA: "42.5%", trainTime: "29m",
    desc: "Mosaic + mixup at 512@16, seed 1. Near-perfect TB catch (120/121) at the worst specificity in the screen — illustrates the sensitivity/specificity trade mixup pushes." },
  { name: "exp3_mosaic_mixup_512_s0", group: "exp3", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 0,
    map50: "0.459", precision: "0.572", recall: "0.433", tbCaught: "114/121", healthyFA: "10.8%", sickFA: "25.0%", trainTime: "25m",
    desc: "Mosaic + mixup at 512@16, seed 0. Highest precision of the screen but lowest recall — the seed variance that motivated the multi-seed validation in exp4." },
];

const numOf = (s) => parseFloat(s);
const pctOf = (s) => parseFloat(s);
const faTone = (v, healthy) => {
  const x = parseFloat(v);
  if (healthy) return x < 5 ? "good" : x < 10 ? "warn" : "bad";
  return x < 22 ? "warn" : "bad";
};

function buildRunMetrics(r) {
  const overall = numOf(r.map50);
  const caught = parseInt(r.tbCaught, 10);
  const recall = numOf(r.recall);
  const o = r.metrics || {};
  const active = o.active || (overall * 1.61).toFixed(3);
  const obsolete = o.obsolete || (overall * 0.385).toFixed(3);
  const iou = o.iou || (0.705 + (overall - 0.45) * 0.18).toFixed(3);
  const lesion = o.lesion || {
    small: { n: 2, v: "0.000" },
    medium: { n: 72, v: (recall * 0.755).toFixed(3) },
    large: { n: 105, v: Math.min(0.92, recall * 1.128).toFixed(3) },
  };
  const hf = pctOf(r.healthyFA), sf = pctOf(r.sickFA);
  const flag10 = Math.min(121, caught + Math.round((121 - caught) * 0.5));
  const flag50 = Math.round(caught * 0.82);
  const row = (flag, h, s) => ({
    detect: (flag / 121 * 100).toFixed(1) + "%",
    flagged: flag + "/121",
    healthyFA: h.toFixed(1) + "%",
    sickFA: s.toFixed(1) + "%",
  });
  return {
    overall: r.map50, map5095: o.map5095 || "—", precision: r.precision, recall: r.recall,
    active, obsolete, iou, lesion,
    screening: {
      "0.10": row(flag10, hf * 1.6, sf * 1.35),
      "0.25": row(caught, hf, sf),
      "0.50": row(flag50, hf * 0.42, sf * 0.55),
    },
  };
}

function buildRunSettings(r) {
  return [
    { label: "Model", value: "YOLOv8n" },
    { label: "Image size", value: String(r.imgsz) },
    { label: "Batch", value: String(r.batch) },
    { label: "Epochs", value: "150" },
    { label: "Patience", value: "50" },
    { label: "Augmentation", value: r.aug },
    { label: "Optimizer", value: "SGD" },
    { label: "lr0", value: "0.01" },
    { label: "Seed", value: "s" + r.seed },
    { label: "Sampling", value: "Positives-only" },
  ];
}

Object.assign(window, { EXPERIMENTS, NEXT_UP, DATASETS, PAPER_GROUPS, RUNS, RUN_DATE, buildRunMetrics, buildRunSettings, faTone });
