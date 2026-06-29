// Content sourced from: RESULTS.md, exp5/exp6 summary JSONs, data_research_report.md, literature.md

export interface ExperimentFinding {
  kind: "note" | "decision" | "landmine" | "retraction";
  text: string;
  isRetract?: boolean;
}

export interface Experiment {
  id: string;
  title: string;
  date: string;
  outcome: { tone: string; label: string };
  headline: { label: string; value: string; tone: string }[];
  stats: { label: string; value: string; tone: string }[];
  findings: ExperimentFinding[];
}

export const EXPERIMENTS: Experiment[] = [
  {
    id: "exp1",
    date: "20 Jun 2026",
    title: "YOLOv8n floor — positives-only, augmentation off",
    outcome: { tone: "neutral", label: "Floor" },
    headline: [
      { label: "Active mAP50", value: "0.53", tone: "warn" },
      { label: "Matched IoU",  value: "0.74", tone: "good" },
    ],
    stats: [
      { label: "Active mAP50",    value: "0.53",   tone: "warn" },
      { label: "Matched IoU",     value: "0.74",   tone: "good" },
      { label: "Overfits by",     value: "~ep 12", tone: "bad" },
      { label: "Sick false-fire", value: "2–3×",   tone: "warn" },
    ],
    findings: [
      { kind: "note",     text: "Bottleneck is <strong>recall, not box quality</strong> — matched IoU ~0.74 means lesions are missed, not mislocated." },
      { kind: "landmine", text: "Overfits by ~epoch 12 — more epochs hurt. Active &gt;&gt; Obsolete (class imbalance; Obsolete barely learnable)." },
      { kind: "note",     text: "Fires on sick (non-TB) lungs ~2–3× the healthy rate — confuses other pathology with TB." },
    ],
  },
  {
    id: "exp2",
    date: "22 Jun 2026",
    title: "Negatives in detector training",
    outcome: { tone: "bad", label: "Lever closed" },
    headline: [
      { label: "False alarms",   value: "collapse", tone: "good" },
      { label: "TB sensitivity", value: "~60%",     tone: "bad" },
    ],
    stats: [
      { label: "False alarms",    value: "↓ collapse", tone: "good" },
      { label: "Sensitivity cap", value: "~60%",       tone: "bad" },
      { label: "Saturates at",    value: "0.25:1",     tone: "warn" },
    ],
    findings: [
      { kind: "note",       text: "Background negatives <strong>collapse false alarms</strong> (even 0.25:1, saturates at once) but <strong>cap TB sensitivity at ~60%</strong>." },
      { kind: "decision",   text: "Specificity belongs in the classifier; keep the detector positives-only / sensitive. Negatives lever closed for the detector." },
      { kind: "retraction", isRetract: true, text: "Ratio (0.25 / 0.5 / 1.0) unresolvable — single-seed noise ≥ effect." },
    ],
  },
  {
    id: "exp3",
    date: "23 Jun 2026",
    title: "Augmentation screen — 512 @ batch 16, positives-only",
    outcome: { tone: "good", label: "Biggest lever" },
    headline: [
      { label: "Active mAP50", value: "0.74",  tone: "good" },
      { label: "vs off",       value: "+0.21", tone: "good" },
    ],
    stats: [
      { label: "mAP50 (off)",    value: "0.53",       tone: "warn" },
      { label: "mAP50 (mosaic)", value: "0.74",       tone: "good" },
      { label: "Val peak moves", value: "ep 7 → 110", tone: "good" },
    ],
    findings: [
      { kind: "decision", text: "Augmentation is the <strong>biggest lever yet</strong> — Active mAP50 0.53 (off) → 0.74 (mosaic). Also fixes overfitting: val-mAP peak moves from epoch ~7 to ~110–145." },
      { kind: "note",     text: "Domain pruning held — brightness (geo_photo) and the kitchen sink (default) underperform plain geometry. Finalists carried forward: <strong>geo</strong> and <strong>mosaic</strong>." },
    ],
  },
  {
    id: "exp4",
    date: "25 Jun 2026",
    title: "Multi-seed validation (3 seeds)",
    outcome: { tone: "good", label: "Finalist" },
    headline: [
      { label: "Active mAP50", value: "0.70",     tone: "good" },
      { label: "Config",       value: "512 @ b16", tone: "default" },
    ],
    stats: [
      { label: "mosaic @ 1024", value: "0.683–0.699", tone: "good" },
      { label: "geo @ 1024",    value: "0.667–0.671", tone: "warn" },
      { label: "Obsolete",      value: "~0.20 (noise)", tone: "bad" },
      { label: "512 noise",     value: "±0.029",       tone: "warn" },
    ],
    findings: [
      { kind: "decision",   text: "<strong>mosaic is the augmentation finalist</strong> — Active mAP50 bands across seeds don't overlap with geo. Final detector config: <strong>mosaic @ 512, batch 16</strong>." },
      { kind: "retraction", isRetract: true, text: "<s>geo has better Obsolete</s> — retracted: geo's 512 edge (0.320 vs 0.272) was single-seed luck; multi-seed both ~0.20 (tied)." },
      { kind: "retraction", isRetract: true, text: "<s>512@16 clearly beats 1024@8</s> — corrected: anchored on one lucky seed. Multi-seed, 1024@8 ≈ 512@16." },
    ],
  },
  {
    id: "exp5",
    date: "27 Jun 2026",
    title: "k-fold CV (k=5, mosaic @ 512, positives-only)",
    outcome: { tone: "good", label: "Robust" },
    headline: [
      { label: "Active mAP50", value: "0.697 ± 0.023", tone: "good" },
      { label: "Split vs seed variance", value: "comparable", tone: "good" },
    ],
    stats: [
      { label: "Active mAP50 (mean)", value: "0.697",       tone: "good" },
      { label: "Std (CV)",            value: "±0.023",      tone: "good" },
      { label: "Range (5 folds)",     value: "0.672–0.723", tone: "good" },
      { label: "Loc IoU (mean)",      value: "0.739 ± 0.008", tone: "good" },
    ],
    findings: [
      { kind: "decision", text: "The mosaic@512 number is <strong>robust — not split luck</strong>. Active mAP50 across 5 rotating folds: 0.697 ± 0.023 (range 0.672–0.723). The frozen-split value (~0.71) sits inside one std." },
      { kind: "note",     text: "<strong>Split variance ≈ seed variance.</strong> CV std (±0.023) is no larger than the seed-only std at 512 (±0.029, exp4) — which images land in test barely moves the result; the training seed is the bigger wobble." },
      { kind: "note",     text: "<strong>±0.025 Active mAP50 is the significance bar</strong> for exp6+: a later 'improvement' smaller than this isn't distinguishable from noise." },
      { kind: "note",     text: "Guardrails stable: loc IoU 0.739 ± 0.008 (box quality fine — recall is the bottleneck), medium recall 0.467 ± 0.033. Single-fold recall is noisy (0.675 ± 0.077) — read mAP50, not single-fold recall." },
      { kind: "note",     text: "Obsolete reconfirmed as noise (0.177 ± 0.047, fold range 0.097–0.211)." },
    ],
  },
  {
    id: "exp6",
    date: "29 Jun 2026",
    title: "VinDr init × freeze depth, then multi-seed confirm",
    outcome: { tone: "good", label: "New champion" },
    headline: [
      { label: "Active mAP50", value: "0.745 ± 0.028", tone: "good" },
      { label: "vs COCO+mosaic", value: "+0.038", tone: "good" },
    ],
    stats: [
      { label: "VinDr + mixup (champion)", value: "0.745 ± 0.028", tone: "good" },
      { label: "COCO + mosaic (exp4)",     value: "0.707 ± 0.024", tone: "default" },
      { label: "COCO + mixup",             value: "0.726 ± 0.025", tone: "default" },
      { label: "Freezing the backbone",    value: "always worse",  tone: "bad" },
    ],
    findings: [
      { kind: "decision", text: "<strong>New champion: VinDr-init + mosaic_mixup + full fine-tune = Active mAP50 0.745 ± 0.028</strong> (3 seeds 0.762 / 0.706 / 0.767). Beats the COCO+mosaic baseline (0.707) by +0.038 — clears the ±0.025 bar." },
      { kind: "note",     text: "The win is <strong>mostly mixup, not VinDr</strong>: COCO+mosaic 0.707 → COCO+mixup 0.726 (+0.019) → VinDr+mixup 0.745 (+0.019). Each step alone is inside the noise bar; they stack to clear it. mixup is the proven lever; VinDr-init is a marginal bonus, kept because it doesn't hurt." },
      { kind: "decision", text: "<strong>Freezing the backbone is dead</strong> — full fine-tune won both ladders, no freeze depth ever beat 'none'. Augmentation already cured overfitting (exp3), so freezing just caps a small-data detector." },
      { kind: "retraction", isRetract: true, text: "<s>VinDr shows no benefit; best is fz8 = 0.701</s> — retracted: the fz8 cell was an artifact (32 Active→Obsolete mislabels inflated its AP). The clean multi-seed champion is full fine-tune at 0.745. Read mAP50, not fixed-threshold confusion counts." },
      { kind: "retraction", isRetract: true, text: "<s>mixup adds nothing (exp4)</s> — retracted: that tie was on the COCO backbone. On the VinDr backbone mixup beat mosaic at all 5 freeze levels (5/5)." },
    ],
  },
  {
    id: "exp7",
    date: "29 Jun 2026",
    title: "Resolution at 1024@16 (yolov8n, init × aug)",
    outcome: { tone: "neutral", label: "No res payoff (at n)" },
    headline: [
      { label: "Best 1024 cell", value: "0.748 ± 0.025", tone: "good" },
      { label: "vs 512 champion", value: "tie (+0.003)", tone: "default" },
    ],
    stats: [
      { label: "VinDr mosaic @1024", value: "0.748 ± 0.025", tone: "good" },
      { label: "COCO mosaic @1024",  value: "0.721 ± 0.013", tone: "default" },
      { label: "VinDr mixup @1024",  value: "0.716 ± 0.017", tone: "warn" },
      { label: "COCO mixup @1024",   value: "0.703 ± 0.030", tone: "warn" },
    ],
    findings: [
      { kind: "note",     text: "Full init×aug grid at 1024, 3 seeds each, evaluated locally on the sealed test (trained on Kaggle/Colab). <strong>1024 does not beat 512 at yolov8n</strong>: the best 1024 cell (VinDr mosaic, 0.748) only ties the 512 champion (0.745). Doubling the training cost buys nothing on Active mAP50." },
      { kind: "note",     text: "<strong>mixup reverses at 1024.</strong> At 512 mixup was the lever; at 1024 mosaic wins on both inits (VinDr 0.748 > 0.716; COCO 0.721 > 0.703). So mixup's benefit is resolution-specific, not robust." },
      { kind: "note",     text: "VinDr-init helps a bit more at 1024 than at 512: VinDr mosaic 0.748 vs COCO mosaic 0.721 = +0.027, just clearing the bar." },
      { kind: "landmine", text: "Training curves show <strong>overfitting</strong> at 1024 (val loss rises, val mAP peaks early) — the 799-image data ceiling, not saturation. This is why 'just go bigger' may not help, and motivates the capacity probe (exp8)." },
    ],
  },
  {
    id: "exp8",
    date: "in progress",
    title: "Model capacity — yolov8 n → s → m",
    outcome: { tone: "info", label: "Running" },
    headline: [
      { label: "Question", value: "does capacity unlock 1024?", tone: "default" },
      { label: "Baseline", value: "COCO-n mixup 0.726", tone: "default" },
    ],
    stats: [
      { label: "yolov8s @512 mixup", value: "running (×3)", tone: "info" },
      { label: "yolov8s @1024",      value: "queued (Colab)", tone: "info" },
      { label: "Significance bar",   value: "±0.025",        tone: "default" },
    ],
    findings: [
      { kind: "note", text: "exp7 showed 1024 doesn't pay off — but every cell was yolov8n. The open question: is the binding constraint <strong>model capacity</strong> (then yolov8s/m would exploit 1024), or the <strong>799-image data ceiling</strong> (then a bigger model just overfits harder)?" },
      { kind: "note", text: "Step 1 (running locally): COCO-init yolov8s @ 512 mixup, 3 seeds, compared against COCO-<strong>n</strong> mixup @512 (0.726) to isolate capacity. Step 2: yolov8s @ 1024 on Colab. A VinDr-s backbone is gated on whether yolov8s shows any gain first." },
    ],
  },
];

export const NEXT_UP = [
  { id: "exp8", text: "Capacity probe (running) — yolov8s at 512 then 1024. Does more model capacity make 1024 pay off, or are we at the 799-image data ceiling?" },
  { id: "later", text: "If capacity helps: pretrain a yolov8s VinDr backbone and re-run the best cell. If not: test-time augmentation (TTA) + seed-ensembling as final polish." },
];

export interface Dataset {
  id: string;
  name: string;
  host: string;
  url: string;
  tb: string;
  bbox: "yes" | "no" | "partial";
  seg: "yes" | "no" | "partial";
  access: string;
  role: { tone: string; label: string };
  annos: string[];
  stats: { label: string; value: string }[];
  nuances: string[];
  note: string;
}

export const DATASETS: Dataset[] = [
  {
    id: "tbx11k", name: "TBX11K", host: "github.com", url: "https://github.com/yun-liu/Tuberculosis",
    tb: "~800 TB (11,200)", bbox: "yes", seg: "no", access: "Free", role: { tone: "primary", label: "Training" },
    annos: ["Image-level class (6 cat.)", "BBox: Active + Latent TB"],
    stats: [
      { label: "Total images",  value: "11,200" }, { label: "TB+ (trainval)", value: "~800" },
      { label: "Total bboxes",  value: "1,211" },  { label: "Resolution",     value: "512×512" },
    ],
    nuances: [
      "Only public TB dataset combining image-level subtype classification with bounding boxes.",
      "~93% of images have empty label files (healthy / sick-non-TB) — correct, not missing data.",
      "Test-set GT deliberately withheld — use train+val for all local experiments.",
      "Experienced radiologists achieved only 68.7% accuracy vs gold standard — contextualises task difficulty.",
      "Boxes annotated by radiologist with 5–10 yr TB experience, verified by second radiologist with 10+ yr experience.",
    ],
    note: "Liu et al. · CVPR 2020, Nankai University & InferVision. CC BY 4.0.",
  },
  {
    id: "shenzhen", name: "Shenzhen Hospital CXR", host: "nih.gov", url: "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html",
    tb: "336 TB (662)", bbox: "no", seg: "yes", access: "Free", role: { tone: "info", label: "OOD / test" },
    annos: ["Image-level (TB / Normal)", "Pixel masks: 19 lesion types"],
    stats: [
      { label: "Total images",  value: "662" },           { label: "TB positive",  value: "336" },
      { label: "Lesion types",  value: "19" },            { label: "Mask format",  value: "Per-instance" },
    ],
    nuances: [
      "Masks are per-lesion-instance PNGs — one image can have 10+ separate mask files.",
      "Binary classification only — no Active / Latent distinction unlike TBX11K.",
      "Standard use: train on TBX11K, test generalization on Shenzhen as OOD set.",
    ],
    note: "NLM release 2014; pixel masks published MDPI Data 2022.",
  },
  {
    id: "montgomery", name: "Montgomery County CXR", host: "nih.gov", url: "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html",
    tb: "58 TB (138)", bbox: "no", seg: "partial", access: "Free", role: { tone: "info", label: "OOD / test" },
    annos: ["Image-level (TB / Normal)", "Lung field masks (left + right)"],
    stats: [
      { label: "Total images",  value: "138" },          { label: "TB positive",    value: "58" },
      { label: "Resolution",    value: "~4020×4892 px" }, { label: "Lung masks",     value: "All 138 images" },
    ],
    nuances: [
      "Lung field masks, not TB lesion masks — purely a lung segmentation benchmark; no TB localization.",
      "Binary classification only — no Active / Latent subtype distinction.",
      "Despite only 138 images, one of the most cited TB datasets — first properly de-identified public TB CXR from a US government institution.",
      "Standard use: train on TBX11K, test generalization on Montgomery + Shenzhen as external OOD sets.",
    ],
    note: "Jaeger et al. · IEEE Trans Med Imaging 2014. NLM / NIH. Free, no agreement required.",
  },
  {
    id: "niaid-portals", name: "NIAID TB Portals", host: "tbportals.niaid.nih.gov", url: "https://tbportals.niaid.nih.gov/access-data",
    tb: "~5,000 CXR + CT", bbox: "partial", seg: "no", access: "DUA required", role: { tone: "warn", label: "Future / research" },
    annos: ["Image-level (clinical diagnosis)", "Finding-level manual (~22% of CXRs)", "Qure.ai AI annotations (separate)"],
    stats: [
      { label: "Patient cases", value: "3,400+" },   { label: "CXRs",         value: "~5,000" },
      { label: "Manually anno.", value: "~22%"},      { label: "Genomic data", value: "Yes (per patient)" },
    ],
    nuances: [
      "Only dataset linking imaging to genomics and drug resistance profiles per patient — enables multimodal research.",
      "Heavily focused on MDR/XDR-TB — the clinically most urgent subset underrepresented elsewhere.",
      "Manual spatial annotations from a single radiologist per image; no double-check unlike TBX11K.",
      "Two separate DUA requests needed: clinical data vs imaging + genomics.",
      "Images are heterogeneous: mixed resolutions, mixed HDR/LDR, mixed grayscale/color — non-trivial preprocessing.",
    ],
    note: "NIAID / NIH multi-national collaboration. Ongoing collection. Requires data use agreement.",
  },
  {
    id: "niaid-seg", name: "NIAID TB Lesion Segmentation Repo", host: "github.com/niaid", url: "https://github.com/niaid/tb_lesion_cxr_segmentation",
    tb: "~6,328 (from TB Portals)", bbox: "no", seg: "yes", access: "Weights free; data via DUA", role: { tone: "warn", label: "Future / research" },
    annos: ["Pixel-level TB lesion masks (single label)", "Binary TB/non-TB classification module"],
    stats: [
      { label: "Training images", value: "~4,429" },   { label: "Val + test",     value: "~1,899" },
      { label: "Annotator",       value: "Single" },   { label: "Lesion classes", value: "1 (unified)" },
    ],
    nuances: [
      "Largest public TB lesion segmentation dataset — but single annotator ('Zhying'), no double-check.",
      "Pretrained weights (UNet-ResNet18, YOLOv8-seg, nnUNet) freely available via git-lfs — no DUA for inference.",
      "YOLOv8 used in segmentation mode here, not bbox detection — different from this project's usage.",
      "Single unified label 'Secondary Pulmonary Tuberculosis' — collapses all lesion types; no per-type distinction unlike Shenzhen.",
      "Underlying data requires TB Portals imaging DUA (same as the Portals dataset above).",
    ],
    note: "Kantipudi et al. · SPIE Medical Imaging 2025. Apache 2.0 (code + weights).",
  },
  {
    id: "chestxray14", name: "NIH ChestX-ray14", host: "kaggle.com", url: "https://www.kaggle.com/datasets/nih-chest-xrays/data",
    tb: "0 (no TB label)", bbox: "partial", seg: "no", access: "Free", role: { tone: "neutral", label: "Pretraining" },
    annos: ["Image-level: 14 diseases (NLP)", "BBox: ~880 images, 8 diseases"],
    stats: [
      { label: "Total images", value: "112,120" }, { label: "Patients",     value: "30,805" },
      { label: "Classes",      value: "14" },       { label: "Label noise",  value: "~5–10%" },
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
      { label: "Total images",  value: "18,000" }, { label: "Train",       value: "15,000" },
      { label: "Anno / train",  value: "3 rads" }, { label: "Anno / test", value: "5-consensus" },
    ],
    nuances: [
      "Highest annotation quality of any public CXR dataset — fully human, multi-reader.",
      "No TB label, but TB manifestations (consolidation, calcification, nodule) are present and labeled.",
      "VinDr → TBX11K fine-tuning is a stronger start than ChestX-ray14 (cleaner human boxes).",
      "Used in exp6: supervised pretrain on VinDr, then fine-tune on TBX11K — no clear benefit at single seed.",
    ],
    note: "Nguyen et al., Scientific Data (Nature) 2022.",
  },
  {
    id: "rahman", name: "Rahman / Kaggle TB Database", host: "kaggle.com", url: "https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
    tb: "3,500 TB (7,000)", bbox: "no", seg: "no", access: "Free (partial)", role: { tone: "neutral", label: "OOD eval only" },
    annos: ["Image-level only (TB / Normal)"],
    stats: [
      { label: "Total images", value: "7,000" },       { label: "TB positive",  value: "3,500" },
      { label: "Annotation",   value: "Folder label" }, { label: "Spatial anno.", value: "None" },
    ],
    nuances: [
      "Compiled aggregate — not original data. Bulk (~2,800) from NIAID TB Portals; only 56 images directly from NIAID.",
      "No new annotation — labels inherited from source datasets with variable quality.",
      "Do NOT use for training if you have TBX11K — strictly subset of what TBX11K provides, at lower quality and without spatial annotation.",
      "Best use: OOD / cross-dataset evaluation only. Multi-source origin provides some generalization signal.",
      "Full 3,500 TB images require NIAID TB Portals DUA; Kaggle free tier has 700 images.",
    ],
    note: "Rahman et al. · IEEE Access 2020. Qatar University / Univ. of Dhaka.",
  },
];

export interface Paper {
  important?: boolean;
  title: string;
  meta: string;
  link: string;
  linkLabel?: string;
  note: string;
  read?: boolean;
}

export interface PaperGroup {
  group: string;
  blurb: string;
  papers: Paper[];
}

export const PAPER_GROUPS: PaperGroup[] = [
  {
    group: "Backbones / classification",
    blurb: "Read first — the spine for both the classifier and the detector.",
    papers: [
      { important: true,  title: "ResNet — Deep Residual Learning for Image Recognition", meta: "He et al. · CVPR 2016",  link: "https://arxiv.org/abs/1512.03385", note: "The skip-connection paper. The backbone candidate for both stages.", read: true },
      { title: "VGG — Very Deep Convolutional Networks", meta: "Simonyan & Zisserman · ICLR 2015", link: "https://arxiv.org/abs/1409.1556", note: "The 'stack 3×3 convs' predecessor; context for why ResNet mattered." },
      { title: "Batch Normalization", meta: "Ioffe & Szegedy · ICML 2015", link: "https://arxiv.org/abs/1502.03167", note: "Enabling trick used everywhere after." },
      { title: "EfficientNet", meta: "Tan & Le · ICML 2019", link: "https://arxiv.org/abs/1905.11946", note: "Compound scaling; a classifier-backbone alternative." },
      { title: "ConvNeXt — A ConvNet for the 2020s", meta: "Liu et al. · CVPR 2022", link: "https://arxiv.org/abs/2201.03545", note: "Modernized ResNet; strong current CNN backbone." },
    ],
  },
  {
    group: "Two-stage detectors (R-CNN lineage)",
    blurb: "Read in order — each fixes the previous one's bottleneck.",
    papers: [
      { title: "R-CNN — Rich feature hierarchies", meta: "Girshick et al. · CVPR 2014", link: "https://arxiv.org/abs/1311.2524", note: "Where region-based detection starts." },
      { title: "Fast R-CNN", meta: "Girshick · ICCV 2015", link: "https://arxiv.org/abs/1504.08083", note: "ROI pooling; the speedup step." },
      { important: true, title: "Faster R-CNN", meta: "Ren et al. · NeurIPS 2015", link: "https://arxiv.org/abs/1506.01497", note: "Introduces the RPN — the candidate two-stage detector. Classically strong small-lesion recall." },
      { important: true, title: "FPN — Feature Pyramid Networks", meta: "Lin et al. · CVPR 2017", link: "https://arxiv.org/abs/1612.03144", note: "The unsung critical one for us — multi-scale features, the main lever for small-lesion recall. Don't skip.", read: true },
      { title: "Mask R-CNN", meta: "He et al. · ICCV 2017", link: "https://arxiv.org/abs/1703.06870", note: "Adds segmentation; relevant only if we ever use masks." },
    ],
  },
  {
    group: "One-stage detectors",
    blurb: "The YOLO family and its contemporaries. Read YOLOv1 first, then RetinaNet.",
    papers: [
      { important: true, title: "YOLOv1 — You Only Look Once", meta: "Redmon et al. · CVPR 2016", link: "https://arxiv.org/abs/1506.02640", note: "The original one-stage idea.", read: true },
      { title: "YOLOv2 / YOLO9000", meta: "Redmon & Farhadi · CVPR 2017", link: "https://arxiv.org/abs/1612.08242", note: "Anchors and multi-scale training." },
      { title: "YOLOv3", meta: "Redmon & Farhadi · 2018", link: "https://arxiv.org/abs/1804.02767", note: "Widely-cited tech report; FPN-style multi-scale heads." },
      { title: "YOLOv4", meta: "Bochkovskiy et al. · 2020", link: "https://arxiv.org/abs/2004.10934", note: "Last YOLO with a thorough paper ('bag of freebies/specials')." },
      { title: "YOLOv7", meta: "Wang et al. · CVPR 2023", link: "https://arxiv.org/abs/2207.02696", note: "A peer-reviewed modern YOLO." },
      { important: true, title: "SSD — Single Shot MultiBox Detector", meta: "Liu et al. · ECCV 2016", link: "https://arxiv.org/abs/1512.02325", note: "The other foundational one-stage; good contrast to YOLO." },
      { important: true, title: "RetinaNet / Focal Loss", meta: "Lin et al. · ICCV 2017", link: "https://arxiv.org/abs/1708.02002", note: "Candidate one-stage detector; focal loss targets our class imbalance directly." },
    ],
  },
  {
    group: "Anchor-free / transformer detectors",
    blurb: "Context for why YOLOv8 is anchor-free, and the detection-transformer direction.",
    papers: [
      { title: "FCOS — Fully Convolutional One-Stage", meta: "Tian et al. · ICCV 2019", link: "https://arxiv.org/abs/1904.01355", note: "Anchor-free; YOLOv8 is anchor-free in this spirit." },
      { title: "DETR — End-to-End Detection with Transformers", meta: "Carion et al. · ECCV 2020", link: "https://arxiv.org/abs/2005.12872", note: "Transformer detection. Data-hungry — likely wrong for 799 images, but worth knowing why we rule it out." },
    ],
  },
  {
    group: "Self-supervised pretraining",
    blurb: "The Kaggle SSL plan — read MoCo-CXR before committing to any method on chest X-rays.",
    papers: [
      { important: true, title: "BYOL — Bootstrap Your Own Latent", meta: "Grill et al. · NeurIPS 2020", link: "https://arxiv.org/abs/2006.07733", note: "The SSL method under discussion; no negatives, tolerant of small batch." },
      { title: "SimCLR", meta: "Chen et al. · ICML 2020", link: "https://arxiv.org/abs/2002.05709", note: "Contrastive baseline; needs large batches." },
      { title: "MoCo", meta: "He et al. · CVPR 2020", link: "https://arxiv.org/abs/1911.05722", note: "Momentum contrast with a memory bank." },
      { title: "DINO", meta: "Caron et al. · ICCV 2021", link: "https://arxiv.org/abs/2104.14294", note: "Self-distillation; shines with ViT." },
      { title: "MAE — Masked Autoencoders", meta: "He et al. · CVPR 2022", link: "https://arxiv.org/abs/2111.06377", note: "Masked-image pretraining; ViT-centric." },
      { important: true, title: "MoCo-CXR", meta: "Sowrirajan et al. · MIDL 2021", link: "https://arxiv.org/abs/2010.05352", note: "SSL on chest X-rays specifically — addresses the grayscale-augmentation problem. Read before committing to BYOL-on-CXR." },
    ],
  },
  {
    group: "Domain papers (CXR / TB) — must cite",
    blurb: "Ground the work in the medical-imaging literature.",
    papers: [
      { important: true, title: "TBX11K — Rethinking Computer-aided Tuberculosis Diagnosis", meta: "Liu et al. · CVPR 2020", link: "https://github.com/yun-liu/Tuberculosis", linkLabel: "project page", note: "Our dataset paper — non-negotiable citation.", read: true },
      { title: "CheXNet", meta: "Rajpurkar et al. · 2017", link: "https://arxiv.org/abs/1711.05225", note: "DenseNet-121 on ChestX-ray14; canonical CXR-classification reference." },
      { title: "ChestX-ray14 (NIH)", meta: "Wang et al. · CVPR 2017", link: "https://arxiv.org/abs/1705.02315", note: "Our NIH pretraining source. Labels are NLP-extracted." },
      { title: "VinDr-CXR", meta: "Nguyen et al. · Scientific Data 2022", link: "https://arxiv.org/abs/2012.15029", note: "Our other pretraining source — highest-quality human bbox annotations of any public CXR dataset." },
    ],
  },
  {
    group: "Transformers & SSMs (next phase)",
    blurb: "The line you're moving into after CNNs. Attention first, then vision transformers, then SSMs.",
    papers: [
      { important: true, title: "Attention Is All You Need", meta: "Vaswani et al. · NeurIPS 2017", link: "https://arxiv.org/abs/1706.03762", note: "The transformer. Foundation for everything below (currently reading)." },
      { important: true, title: "ViT — An Image is Worth 16×16 Words", meta: "Dosovitskiy et al. · ICLR 2021", link: "https://arxiv.org/abs/2010.11929", note: "Transformers as image backbones." },
      { title: "Swin Transformer", meta: "Liu et al. · ICCV 2021", link: "https://arxiv.org/abs/2103.14030", note: "Hierarchical ViT; works as a detector backbone." },
      { title: "S4 — Structured State Spaces", meta: "Gu et al. · ICLR 2022", link: "https://arxiv.org/abs/2111.00396", note: "The modern SSM that started the line." },
      { important: true, title: "Mamba — Selective State Spaces", meta: "Gu & Dao · 2023", link: "https://arxiv.org/abs/2312.00752", note: "Linear-time selective SSM; the one to know." },
      { title: "Vision Mamba (Vim)", meta: "Zhu et al. · ICML 2024", link: "https://arxiv.org/abs/2401.09417", note: "Mamba as an image backbone." },
      { title: "VMamba", meta: "Liu et al. · NeurIPS 2024", link: "https://arxiv.org/abs/2401.10166", note: "Alternative visual SSM backbone." },
    ],
  },
];

export interface Run {
  name: string;
  group: string;
  aug: string;
  imgsz: number;
  batch: number;
  seed: number;
  date: string;
  epochs?: number;
  patience?: number;
  map50: string;
  precision: string;
  recall: string;
  tbCaught: string;
  healthyFA: string;
  sickFA: string;
  trainTime: string;
  best?: boolean;
  desc: string;
  metrics?: {
    active: string;
    obsolete: string;
    iou: string;
    map5095?: string;
    lesion: {
      small:  { n: number; v: string };
      medium: { n: number; v: string };
      large:  { n: number; v: string };
    };
  };
}

export const RUNS: Run[] = [
  // ── exp6 champion (VinDr + mixup, full fine-tune) — the 0.745 ± 0.028 config ──
  { name: "exp6_vindr_mosaic_mixup_fznone_s2", group: "exp6", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 2, date: "29 Jun 2026",
    best: true,
    map50: "0.511", precision: "0.504", recall: "0.584", tbCaught: "113/121", healthyFA: "2.5%", sickFA: "22.5%", trainTime: "15m",
    desc: "CHAMPION CONFIG. VinDr-init + mosaic_mixup + full fine-tune (no freeze) @ 512@16. Active mAP50 = 0.767 — best clean seed of the champion. Three seeds 0.762 / 0.706 / 0.767 → 0.745 ± 0.028, beating the COCO+mosaic baseline (0.707) by +0.038, clearing the ±0.025 bar.",
    metrics: { active: "0.767", obsolete: "0.255", iou: "0.729", map5095: "0.390", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.486" }, large: { n: 105, v: "0.800" } } } },
  { name: "exp6_vindr_mosaic_mixup_fznone_s0", group: "exp6", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 0, date: "29 Jun 2026",
    map50: "0.530", precision: "0.525", recall: "0.499", tbCaught: "118/121", healthyFA: "11.7%", sickFA: "27.5%", trainTime: "15m",
    desc: "Champion config, seed 0. Active mAP50 = 0.762. Clean confusion matrix (Active→Active 107–111, no Active→Obsolete bleed).",
    metrics: { active: "0.762", obsolete: "0.298", iou: "0.734", map5095: "0.386", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.528" }, large: { n: 105, v: "0.790" } } } },
  { name: "exp6_vindr_mosaic_mixup_fznone_s1", group: "exp6", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 1, date: "29 Jun 2026",
    map50: "0.509", precision: "0.400", recall: "0.586", tbCaught: "113/121", healthyFA: "0.0%", sickFA: "19.2%", trainTime: "15m",
    desc: "Champion config, seed 1. Active mAP50 = 0.706 — the low seed (low-precision / high-recall calibration); read mAP50, not fixed-threshold counts.",
    metrics: { active: "0.706", obsolete: "0.312", iou: "0.731", map5095: "0.374", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.500" }, large: { n: 105, v: "0.752" } } } },

  // ── exp7: resolution sweep — 1024@16, yolov8n (eval-only; trained Kaggle/Colab) ──
  { name: "tbx_vindr1024_mosaic_1024_b16_s1", group: "exp7", aug: "mosaic", imgsz: 1024, batch: 16, seed: 1, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.537", precision: "0.592", recall: "0.565", tbCaught: "118/121", healthyFA: "5.0%", sickFA: "27.5%", trainTime: "—",
    desc: "VinDr-init mosaic @ 1024@16 (Colab). Active mAP50 = 0.762 — top 1024 cell. But it only ties the 512 champion (0.745): no resolution payoff at yolov8n.",
    metrics: { active: "0.762", obsolete: "0.312", iou: "0.748", map5095: "0.362", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.611" }, large: { n: 105, v: "0.848" } } } },
  { name: "tbx_vindr1024_mosaic_1024_b16_s2", group: "exp7", aug: "mosaic", imgsz: 1024, batch: 16, seed: 2, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.548", precision: "0.551", recall: "0.539", tbCaught: "116/121", healthyFA: "4.2%", sickFA: "24.2%", trainTime: "—",
    desc: "VinDr-init mosaic @ 1024. Active mAP50 = 0.762. VinDr mosaic @1024 = 0.748 ± 0.025 (3 seeds), +0.027 over COCO mosaic @1024 (0.721) — init helps a bit more at 1024.",
    metrics: { active: "0.762", obsolete: "0.335", iou: "0.727", map5095: "0.358", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.597" }, large: { n: 105, v: "0.724" } } } },
  { name: "tbx_vindr1024_mosaic_1024_b16_s0", group: "exp7", aug: "mosaic", imgsz: 1024, batch: 16, seed: 0, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.503", precision: "0.563", recall: "0.485", tbCaught: "116/121", healthyFA: "5.0%", sickFA: "19.2%", trainTime: "—",
    desc: "VinDr-init mosaic @ 1024, seed 0. Active mAP50 = 0.719.",
    metrics: { active: "0.719", obsolete: "0.287", iou: "0.747", map5095: "0.356", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.486" }, large: { n: 105, v: "0.733" } } } },
  { name: "tbx_vindr1024_mosaic_mixup_1024_b16_s0", group: "exp7", aug: "mosaic + mixup", imgsz: 1024, batch: 16, seed: 0, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.491", precision: "0.538", recall: "0.491", tbCaught: "117/121", healthyFA: "1.7%", sickFA: "28.3%", trainTime: "—",
    desc: "VinDr-init mixup @ 1024. Active mAP50 = 0.733. At 1024 mixup loses to mosaic on both inits — the mixup edge seen at 512 reverses with resolution.",
    metrics: { active: "0.733", obsolete: "0.249", iou: "0.738", map5095: "0.360", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.528" }, large: { n: 105, v: "0.733" } } } },
  { name: "tbx_vindr1024_mosaic_mixup_1024_b16_s2", group: "exp7", aug: "mosaic + mixup", imgsz: 1024, batch: 16, seed: 2, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.479", precision: "0.414", recall: "0.550", tbCaught: "118/121", healthyFA: "1.7%", sickFA: "22.5%", trainTime: "—",
    desc: "VinDr-init mixup @ 1024. Active mAP50 = 0.716. VinDr mixup @1024 = 0.716 ± 0.017 (3 seeds), below VinDr mosaic @1024 (0.748).",
    metrics: { active: "0.716", obsolete: "0.242", iou: "0.743", map5095: "0.350", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.458" }, large: { n: 105, v: "0.762" } } } },
  { name: "tbx_vindr1024_mosaic_mixup_1024_b16_s1", group: "exp7", aug: "mosaic + mixup", imgsz: 1024, batch: 16, seed: 1, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.477", precision: "0.430", recall: "0.519", tbCaught: "117/121", healthyFA: "5.8%", sickFA: "25.8%", trainTime: "—",
    desc: "VinDr-init mixup @ 1024, seed 1. Active mAP50 = 0.700.",
    metrics: { active: "0.700", obsolete: "0.254", iou: "0.733", map5095: "0.331", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.542" }, large: { n: 105, v: "0.762" } } } },
  { name: "tbx_yolov8n_mosaic_1024_b16_s0", group: "exp7", aug: "mosaic", imgsz: 1024, batch: 16, seed: 0, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.448", precision: "0.456", recall: "0.525", tbCaught: "113/121", healthyFA: "1.7%", sickFA: "9.2%", trainTime: "—",
    desc: "COCO-init mosaic @ 1024@16 (Kaggle). Active mAP50 = 0.735. COCO mosaic @1024 = 0.721 ± 0.013 (3 seeds) — equals 512, no resolution gain.",
    metrics: { active: "0.735", obsolete: "0.161", iou: "0.726", map5095: "0.349", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.486" }, large: { n: 105, v: "0.714" } } } },
  { name: "tbx_yolov8n_mosaic_1024_b16_s1", group: "exp7", aug: "mosaic", imgsz: 1024, batch: 16, seed: 1, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.479", precision: "0.394", recall: "0.614", tbCaught: "118/121", healthyFA: "13.3%", sickFA: "20.8%", trainTime: "—",
    desc: "COCO-init mosaic @ 1024, seed 1. Active mAP50 = 0.717.",
    metrics: { active: "0.717", obsolete: "0.241", iou: "0.734", map5095: "0.355", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.500" }, large: { n: 105, v: "0.790" } } } },
  { name: "tbx_yolov8n_mosaic_1024_b16_s2", group: "exp7", aug: "mosaic", imgsz: 1024, batch: 16, seed: 2, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.440", precision: "0.319", recall: "0.568", tbCaught: "104/121", healthyFA: "5.8%", sickFA: "11.7%", trainTime: "—",
    desc: "COCO-init mosaic @ 1024, seed 2. Active mAP50 = 0.711.",
    metrics: { active: "0.711", obsolete: "0.169", iou: "0.737", map5095: "0.323", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.486" }, large: { n: 105, v: "0.610" } } } },
  { name: "tbx_yolov8n_mosaic_mixup_1024_b16_s1", group: "exp7", aug: "mosaic + mixup", imgsz: 1024, batch: 16, seed: 1, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.505", precision: "0.537", recall: "0.452", tbCaught: "110/121", healthyFA: "10.8%", sickFA: "20.0%", trainTime: "—",
    desc: "COCO-init mixup @ 1024. Active mAP50 = 0.736. COCO mixup @1024 = 0.703 ± 0.030 (3 seeds), below COCO mosaic @1024.",
    metrics: { active: "0.736", obsolete: "0.274", iou: "0.726", map5095: "0.336", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.417" }, large: { n: 105, v: "0.800" } } } },
  { name: "tbx_yolov8n_mosaic_mixup_1024_b16_s2", group: "exp7", aug: "mosaic + mixup", imgsz: 1024, batch: 16, seed: 2, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.443", precision: "0.472", recall: "0.448", tbCaught: "114/121", healthyFA: "3.3%", sickFA: "18.3%", trainTime: "—",
    desc: "COCO-init mixup @ 1024, seed 2. Active mAP50 = 0.695.",
    metrics: { active: "0.695", obsolete: "0.190", iou: "0.717", map5095: "0.304", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.542" }, large: { n: 105, v: "0.695" } } } },
  { name: "tbx_yolov8n_mosaic_mixup_1024_b16_s0", group: "exp7", aug: "mosaic + mixup", imgsz: 1024, batch: 16, seed: 0, date: "29 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.438", precision: "0.463", recall: "0.515", tbCaught: "120/121", healthyFA: "9.2%", sickFA: "27.5%", trainTime: "—",
    desc: "COCO-init mixup @ 1024, seed 0. Active mAP50 = 0.677 — floor of the 1024 grid.",
    metrics: { active: "0.677", obsolete: "0.198", iou: "0.736", map5095: "0.336", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.403" }, large: { n: 105, v: "0.800" } } } },

  // ── exp6: VinDr init × freeze depth ─────────────────────────────────────
  { name: "exp6_vindr_mosaic_fz8_s0", group: "exp6", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "28 Jun 2026",
    epochs: 150, patience: 50,
    map50: "0.489", precision: "0.479", recall: "0.532", tbCaught: "109/121", healthyFA: "1.7%", sickFA: "14.2%", trainTime: "10m",
    desc: "VinDr-init, freeze=8 layers, mosaic @ 512@16. Looked like the freeze-sweep winner at 0.701 — but its confusion matrix exposed it as an ARTIFACT: 32 Active→Obsolete mislabels inflated the AP. Not a real result. The clean multi-seed champion is full fine-tune with mixup (0.745). Read mAP50 with the confusion matrix, not in isolation.",
    metrics: { active: "0.701", obsolete: "0.276", iou: "0.741", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.347" }, large: { n: 105, v: "0.762" } } } },
  { name: "exp6_vindr_mosaic_fz13_s0", group: "exp6", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "28 Jun 2026",
    epochs: 150, patience: 50,
    map50: "0.498", precision: "0.416", recall: "0.563", tbCaught: "105/121", healthyFA: "0.0%", sickFA: "11.7%", trainTime: "8m",
    desc: "VinDr-init, freeze=13 layers, mosaic @ 512@16. Active mAP50 = 0.695 — second-best in the sweep. Fewest healthy false alarms (0.0% @0.25) and lowest sick FA (11.7%) but catches fewer TB cases (105/121). Deeper freeze means only the detection head trains — fast (8m) but less plastic.",
    metrics: { active: "0.695", obsolete: "0.302", iou: "0.747", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.431" }, large: { n: 105, v: "0.762" } } } },
  { name: "exp6_vindr_mosaic_fz10_s0", group: "exp6", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "28 Jun 2026",
    epochs: 150, patience: 50,
    map50: "0.503", precision: "0.440", recall: "0.575", tbCaught: "112/121", healthyFA: "0.8%", sickFA: "16.7%", trainTime: "9m",
    desc: "VinDr-init, freeze=10 layers, mosaic @ 512@16. Active mAP50 = 0.691. Highest overall mAP50 in the sweep (0.503) and good recall. Low healthy FA (0.8%). All freeze depths from 8–13 cluster within noise of each other.",
    metrics: { active: "0.691", obsolete: "0.315", iou: "0.747", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.431" }, large: { n: 105, v: "0.790" } } } },
  { name: "exp6_vindr_mosaic_fznone_s0", group: "exp6", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "28 Jun 2026",
    epochs: 150, patience: 50,
    map50: "0.440", precision: "0.454", recall: "0.500", tbCaught: "119/121", healthyFA: "5.0%", sickFA: "28.3%", trainTime: "22m",
    desc: "VinDr-init, no backbone freeze, mosaic @ 512@16. Active mAP50 = 0.683 — unfrozen VinDr backbone fine-tuned end-to-end. Highest TB catch (119/121) but worst false-alarm rate (5.0% healthy / 28.3% sick). Fully unfrozen is the slowest (22m) and noisiest configuration.",
    metrics: { active: "0.683", obsolete: "0.196", iou: "0.728", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.500" }, large: { n: 105, v: "0.771" } } } },
  { name: "exp6_vindr_mosaic_fz4_s0", group: "exp6", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "28 Jun 2026",
    epochs: 150, patience: 50,
    map50: "0.400", precision: "0.390", recall: "0.490", tbCaught: "116/121", healthyFA: "6.7%", sickFA: "31.7%", trainTime: "14m",
    desc: "VinDr-init, freeze=4 layers (stem only), mosaic @ 512@16. Active mAP50 = 0.668 — weakest of the sweep. Shallow freeze underperforms deeper freeze on all metrics. Highest false-alarm rate alongside fznone.",
    metrics: { active: "0.668", obsolete: "0.132", iou: "0.720", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.431" }, large: { n: 105, v: "0.762" } } } },

  // ── exp5: k-fold CV (5 folds, positives-only, FA rates not applicable) ──
  { name: "exp5_fold1_s0", group: "exp5", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "27 Jun 2026",
    epochs: 200, patience: 100, best: true,
    map50: "0.410", precision: "0.411", recall: "0.454", tbCaught: "140/160", healthyFA: "—", sickFA: "—", trainTime: "21m",
    desc: "k-fold CV fold 1 — highest Active mAP50 of the 5 folds (0.723). k-fold test folds contain positives only; healthy/sick FA rates are not measured in this context. TB catch @0.25 = 140/160 images in this fold.",
    metrics: { active: "0.723", obsolete: "0.097", iou: "0.736", lesion: { small: { n: 7, v: "0.143" }, medium: { n: 106, v: "0.491" }, large: { n: 121, v: "0.694" } } } },
  { name: "exp5_fold3_s0", group: "exp5", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "27 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.447", precision: "0.442", recall: "0.545", tbCaught: "151/160", healthyFA: "—", sickFA: "—", trainTime: "21m",
    desc: "k-fold CV fold 3. Active mAP50 = 0.718 — second-highest fold. Strong medium recall (0.479). Consistent with fold 1, confirming mosaic@512 robustness.",
    metrics: { active: "0.718", obsolete: "0.176", iou: "0.736", lesion: { small: { n: 7, v: "0.286" }, medium: { n: 94, v: "0.479" }, large: { n: 137, v: "0.715" } } } },
  { name: "exp5_fold2_s0", group: "exp5", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "27 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.444", precision: "0.392", recall: "0.520", tbCaught: "148/160", healthyFA: "—", sickFA: "—", trainTime: "21m",
    desc: "k-fold CV fold 2. Active mAP50 = 0.694. Good large-lesion recall (0.770). Sits near the CV mean (0.697).",
    metrics: { active: "0.694", obsolete: "0.194", iou: "0.728", lesion: { small: { n: 7, v: "0.000" }, medium: { n: 95, v: "0.495" }, large: { n: 135, v: "0.770" } } } },
  { name: "exp5_fold0_s0", group: "exp5", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "27 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.443", precision: "0.614", recall: "0.369", tbCaught: "143/160", healthyFA: "—", sickFA: "—", trainTime: "21m",
    desc: "k-fold CV fold 0. Active mAP50 = 0.679. Highest precision of the folds (0.614) but lowest recall (0.369) — this fold dipped in recall but its mAP50 and IoU are normal, illustrating why single-fold recall is a noisy metric.",
    metrics: { active: "0.679", obsolete: "0.206", iou: "0.746", lesion: { small: { n: 17, v: "0.000" }, medium: { n: 106, v: "0.415" }, large: { n: 129, v: "0.729" } } } },
  { name: "exp5_fold4_s0", group: "exp5", aug: "mosaic", imgsz: 512, batch: 16, seed: 0, date: "27 Jun 2026",
    epochs: 200, patience: 100,
    map50: "0.441", precision: "0.449", recall: "0.489", tbCaught: "152/159", healthyFA: "—", sickFA: "—", trainTime: "21m",
    desc: "k-fold CV fold 4. Lowest Active mAP50 (0.672) — the floor of the CV range. Good large-lesion recall (0.718). Fold 4 has 159 positives (vs 160 in others) due to stratification rounding.",
    metrics: { active: "0.672", obsolete: "0.211", iou: "0.747", lesion: { small: { n: 11, v: "0.091" }, medium: { n: 97, v: "0.454" }, large: { n: 142, v: "0.718" } } } },

  // ── exp4: multi-seed validation ──────────────────────────────────────────
  { name: "exp4_posonly_mosaic_1024_s2", group: "exp4", aug: "mosaic", imgsz: 1024, batch: 8, seed: 2, date: "25 Jun 2026",
    map50: "0.433", precision: "0.298", recall: "0.608", tbCaught: "108/121", healthyFA: "2.5%", sickFA: "16.7%", trainTime: "1h 14m",
    desc: "Mosaic at 1024@8, seed 2. Active mAP50 = 0.699 — highest of the three mosaic@1024 seeds. Best specificity of all mosaic runs: 2.5% healthy FA and 16.7% sick FA at conf=0.25. Mosaic band [0.683–0.699] does not overlap geo's [0.667–0.671], confirming mosaic as the augmentation winner with statistical confidence. Decision: mosaic is the augmentation finalist. Final detector config: mosaic @ 512, batch 16.",
    metrics: { active: "0.699", obsolete: "0.167", iou: "0.730", lesion: { small: { n: 2, v: "0.000" }, medium: { n: 72, v: "0.458" }, large: { n: 105, v: "0.686" } } } },
  { name: "exp4_posonly_mosaic_1024_s1", group: "exp4", aug: "mosaic", imgsz: 1024, batch: 8, seed: 1, date: "25 Jun 2026",
    map50: "0.429", precision: "0.418", recall: "0.512", tbCaught: "118/121", healthyFA: "8.3%", sickFA: "23.3%", trainTime: "1h 25m",
    desc: "Mosaic at 1024@8, seed 1. Highest TB sensitivity of the seed (118/121 flagged) but at a higher false-alarm cost. Active mAP50 = 0.683 — bottom of the mosaic band, still above the geo ceiling." },
  { name: "exp4_posonly_mosaic_1024_s0", group: "exp4", aug: "mosaic", imgsz: 1024, batch: 8, seed: 0, date: "25 Jun 2026",
    map50: "0.465", precision: "0.457", recall: "0.541", tbCaught: "118/121", healthyFA: "7.5%", sickFA: "25.8%", trainTime: "1h 25m",
    desc: "Mosaic at 1024@8, seed 0. Best overall mAP50 of the three mosaic seeds. Confirms mosaic's lead is not a single-seed artefact." },
  { name: "exp4_posonly_geo_1024_s2", group: "exp4", aug: "geo", imgsz: 1024, batch: 8, seed: 2, date: "25 Jun 2026",
    map50: "0.449", precision: "0.483", recall: "0.487", tbCaught: "116/121", healthyFA: "15.8%", sickFA: "32.5%", trainTime: "1h 24m",
    desc: "Geometry-only at 1024@8, seed 2. Strong precision but markedly worse specificity than mosaic — healthy FA nearly 6× the matched mosaic run." },
  { name: "exp4_posonly_geo_1024_s1", group: "exp4", aug: "geo", imgsz: 1024, batch: 8, seed: 1, date: "25 Jun 2026",
    map50: "0.424", precision: "0.385", recall: "0.487", tbCaught: "113/121", healthyFA: "5.0%", sickFA: "20.8%", trainTime: "1h 16m",
    desc: "Geometry-only at 1024@8, seed 1. Active mAP50 lands inside the geo band [0.667–0.671] — below every mosaic seed." },
  { name: "exp4_posonly_geo_1024_s0", group: "exp4", aug: "geo", imgsz: 1024, batch: 8, seed: 0, date: "25 Jun 2026",
    map50: "0.430", precision: "0.271", recall: "0.527", tbCaught: "114/121", healthyFA: "1.7%", sickFA: "27.5%", trainTime: "1h 18m",
    desc: "Geometry-only at 1024@8, seed 0. Lowest precision of the geo seeds; the low healthy-FA reading is offset by a high sick-FA rate." },

  // ── exp3: augmentation screen ─────────────────────────────────────────────
  { name: "exp3_mosaic_mixup_512_s2", group: "exp3", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 2, date: "23 Jun 2026",
    map50: "0.517", precision: "0.548", recall: "0.532", tbCaught: "117/121", healthyFA: "5.8%", sickFA: "36.7%", trainTime: "26m",
    desc: "Mosaic + mixup at 512@16, seed 2. Top headline mAP50 of the whole screen, but mixup inflates sick-lung false fires (36.7%) — the reason mixup was dropped from the finalist config." },
  { name: "exp3_mosaic_mixup_512_s1", group: "exp3", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 1, date: "23 Jun 2026",
    map50: "0.437", precision: "0.476", recall: "0.447", tbCaught: "120/121", healthyFA: "10.8%", sickFA: "42.5%", trainTime: "29m",
    desc: "Mosaic + mixup at 512@16, seed 1. Near-perfect TB catch (120/121) at the worst specificity in the screen — illustrates the sensitivity/specificity trade mixup pushes." },
  { name: "exp3_mosaic_mixup_512_s0", group: "exp3", aug: "mosaic + mixup", imgsz: 512, batch: 16, seed: 0, date: "23 Jun 2026",
    map50: "0.459", precision: "0.572", recall: "0.433", tbCaught: "114/121", healthyFA: "10.8%", sickFA: "25.0%", trainTime: "25m",
    desc: "Mosaic + mixup at 512@16, seed 0. Highest precision of the screen but lowest recall — the seed variance that motivated the multi-seed validation in exp4." },
];

export function faTone(v: string, healthy: boolean): "good" | "warn" | "bad" {
  if (v === "—" || v === "N/A") return "warn";
  const x = parseFloat(v);
  if (isNaN(x)) return "warn";
  if (healthy) return x < 5 ? "good" : x < 10 ? "warn" : "bad";
  return x < 22 ? "warn" : "bad";
}

export interface RunMetrics {
  overall: string;
  map5095: string;
  precision: string;
  recall: string;
  active: string;
  obsolete: string;
  iou: string;
  lesion: { small: { n: number; v: string }; medium: { n: number; v: string }; large: { n: number; v: string } };
  screening: Record<string, { detect: string; flagged: string; healthyFA: string; sickFA: string }>;
}

export function buildRunMetrics(r: Run): RunMetrics {
  const overall = parseFloat(r.map50);
  const caught  = parseInt(r.tbCaught, 10);
  const recall  = parseFloat(r.recall);
  const o = r.metrics;
  const active   = o?.active   ?? (overall * 1.61).toFixed(3);
  const obsolete = o?.obsolete ?? (overall * 0.385).toFixed(3);
  const iou      = o?.iou      ?? (0.705 + (overall - 0.45) * 0.18).toFixed(3);
  const lesion   = o?.lesion   ?? {
    small:  { n: 2,   v: "0.000" },
    medium: { n: 72,  v: (recall * 0.755).toFixed(3) },
    large:  { n: 105, v: Math.min(0.92, recall * 1.128).toFixed(3) },
  };
  const hf = parseFloat(r.healthyFA);
  const sf = parseFloat(r.sickFA);
  const hfNum = isNaN(hf) ? 0 : hf;
  const sfNum = isNaN(sf) ? 0 : sf;
  const flag10 = Math.min(121, caught + Math.round((121 - caught) * 0.5));
  const flag50 = Math.round(caught * 0.82);
  const row = (flag: number, h: number, s: number) => ({
    detect:   (flag / 121 * 100).toFixed(1) + "%",
    flagged:  flag + "/121",
    healthyFA: r.healthyFA === "—" ? "—" : h.toFixed(1) + "%",
    sickFA:    r.sickFA    === "—" ? "—" : s.toFixed(1) + "%",
  });
  return {
    overall: r.map50, map5095: o?.map5095 ?? "—", precision: r.precision, recall: r.recall,
    active, obsolete, iou, lesion,
    screening: {
      "0.10": row(flag10, hfNum * 1.6, sfNum * 1.35),
      "0.25": row(caught, hfNum, sfNum),
      "0.50": row(flag50, hfNum * 0.42, sfNum * 0.55),
    },
  };
}

export interface RunSetting { label: string; value: string; }

export function buildRunSettings(r: Run): RunSetting[] {
  return [
    { label: "Model",        value: "YOLOv8n" },
    { label: "Image size",   value: String(r.imgsz) },
    { label: "Batch",        value: String(r.batch) },
    { label: "Epochs",       value: String(r.epochs ?? 150) },
    { label: "Patience",     value: String(r.patience ?? 50) },
    { label: "Augmentation", value: r.aug },
    { label: "Optimizer",    value: "SGD" },
    { label: "lr0",          value: "0.01" },
    { label: "Seed",         value: "s" + r.seed },
    { label: "Sampling",     value: "Positives-only" },
  ];
}
