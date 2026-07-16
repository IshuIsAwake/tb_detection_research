// Reported results per architecture, from the survey in
// docs/TB_Model_Dataset_Metrics.md (itself compiled from "Exhaustive Analysis of
// Deep Learning Architectures for Tuberculosis Detection in Chest Radiography").
//
// These are the numbers the SOURCE STUDIES claim, reproduced as reported. They
// are NOT comparable with each other: the datasets differ in size and difficulty,
// several are private or unnamed, and most are binary classification rather than
// localisation. The page says so loudly rather than ranking them.

export type Family = "CNN" | "Segmentation" | "Detection" | "Transformer" | "SSM" | "Hybrid" | "Commercial";

export interface ArchRow {
  name: string;
  family: Family;
  task: string;
  study: string;
  dataset: string;
  /** Headline metric, as reported. */
  metric: string;
  /** Numeric form of the headline accuracy/AUC for the chart, where one exists. */
  value?: number;
  /** Reported on a private or unnamed dataset — the caveat that matters most. */
  privateData?: boolean;
}

export const ARCH_ROWS: ArchRow[] = [
  {
    name: "AlexNet + GoogLeNet (ensemble)",
    family: "CNN",
    task: "Classification (TB vs normal)",
    study: "Lakhani & Sundaram (2017), Radiology",
    dataset: "Private, 1,007 posteroanterior CXRs",
    metric: "AUC 0.99 · acc 95.3% · sens 92.0% · spec 98.7%",
    value: 95.3,
    privateData: true,
  },
  {
    name: "AlexNet (pretrained)",
    family: "CNN",
    task: "Classification (binary)",
    study: "Transfer-learning study",
    dataset: "Microscopic slide images — not CXR",
    metric: "acc 98.73% · sens 98.59%",
    value: 98.73,
    privateData: true,
  },
  {
    name: "VGG-16",
    family: "CNN",
    task: "Classification (binary)",
    study: "Huang et al. (2025)",
    dataset: "Private, 4,200 CXRs (700 TB+)",
    metric: "acc 99.4% · prec 97.9% · rec 98.6% · AUC 98.25%",
    value: 99.4,
    privateData: true,
  },
  {
    name: "ResNet (multi-class)",
    family: "CNN",
    task: "Classification (normal / viral / bacterial / TB)",
    study: "Multi-class hierarchical studies",
    dataset: "Not specified",
    metric: "acc > 96%",
    value: 96,
    privateData: true,
  },
  {
    name: "ResNet (TBNet, semi-supervised)",
    family: "CNN",
    task: "Classification (binary, semi-supervised)",
    study: "Liu et al.",
    dataset: "100,000+ unlabelled + external test",
    metric: "AUC 0.91 (external test set)",
    value: 91,
  },
  {
    name: "DenseNet-169 + SVM/DNN",
    family: "CNN",
    task: "Classification (binary)",
    study: "Ganapathy et al.",
    dataset: "Three public datasets (unspecified which)",
    metric: "acc 99.91% · prec 99.23% · rec 99.22% · AUC ~99.99%",
    value: 99.91,
    privateData: true,
  },
  {
    name: "MobileNet + AEO",
    family: "CNN",
    task: "Classification + feature selection",
    study: "Sahlol et al. (2020)",
    dataset: "External validation set",
    metric: "acc 94.1% (external validation) · 50,000 → <25 features",
    value: 94.1,
  },
  {
    name: "EfficientNetV2 (multimodal)",
    family: "CNN",
    task: "Classification (multimodal fusion)",
    study: "Multimodal CDS integrations",
    dataset: "CXR + structured patient data",
    metric: "acc > 90%",
    value: 90,
    privateData: true,
  },
  {
    name: "U-Net",
    family: "Segmentation",
    task: "Segmentation (lung fields)",
    study: "Sharma et al. (2023)",
    dataset: "Montgomery + Shenzhen",
    metric: "acc 96.35% · Jaccard 90.38% · Dice 94.88%",
    value: 96.35,
  },
  {
    name: "Attention U-Net + Swin",
    family: "Segmentation",
    task: "Segmentation + classification",
    study: "Visu et al. (2025)",
    dataset: "Not specified",
    metric: "diagnostic acc > 99%",
    value: 99,
    privateData: true,
  },
  {
    name: "Faster R-CNN",
    family: "Detection",
    task: "Object detection (lesion localisation)",
    study: "Xie et al. (2020)",
    dataset: "Shenzhen + Montgomery",
    metric: "AUC 0.977 · acc 92.6%",
    value: 92.6,
  },
  {
    name: "YOLO (general)",
    family: "Detection",
    task: "Object detection (real-time)",
    study: "Various contemporary implementations",
    dataset: "Not specified",
    metric: "~68% mAP — speed over precision",
    value: 68,
  },
  {
    name: "YOLO-CXR (Selective Feature Fusion)",
    family: "Detection",
    task: "Object detection (small / early lesions)",
    study: "Specialised small-lesion variant",
    dataset: "Not specified",
    metric: "Improved small-lesion detection (no headline metric)",
  },
  {
    name: "ViT + CLAHE (hybrid)",
    family: "Transformer",
    task: "Classification (binary)",
    study: "Hybrid transformer study",
    dataset: "Not specified",
    metric: "AP 99.90% · acc up to 99.84% · F1 99.71%",
    value: 99.84,
    privateData: true,
  },
  {
    name: "Enhanced Swin (EnSTrans) + RPN",
    family: "Transformer",
    task: "Classification + segmentation",
    study: "Visu et al. (2025)",
    dataset: "Not specified",
    metric: "acc 99.05%",
    value: 99.05,
    privateData: true,
  },
  {
    name: "TB-Net (self-attention CNN)",
    family: "Transformer",
    task: "Classification (binary)",
    study: "Wong et al. (2021)",
    dataset: "Multi-national screening cohorts",
    metric: "acc 99.86% · sens 100%",
    value: 99.86,
    privateData: true,
  },
  {
    name: "Vision Mamba CGSM",
    family: "SSM",
    task: "Multi-class + active vs inactive TB",
    study: "Jeon et al. (2026)",
    dataset: "Normal / Pneumonia / Active TB / Inactive TB",
    metric: "multi-class acc 92.96% · active-vs-inactive spec 97.04% · Youden 79.55%",
    value: 92.96,
  },
  {
    name: "Vision Mamba (general)",
    family: "SSM",
    task: "Classification (efficiency-focused)",
    study: "Mamba vs CNN/ViT comparison",
    dataset: "Not specified",
    metric: "~80% less GPU memory than ViT at comparable accuracy",
  },
  {
    name: "RepViT-CXR",
    family: "Hybrid",
    task: "Classification (TB + pneumonia)",
    study: "arXiv preprint",
    dataset: "Multiple (unspecified)",
    metric: "New SOTA claimed (no numbers given)",
  },
  {
    name: "GhostNet + MobileViT",
    family: "Hybrid",
    task: "Classification (efficiency-focused)",
    study: "Owda et al.",
    dataset: "Not named",
    metric: "acc > 99% · 7.73M params · 282.11M FLOPs",
    value: 99,
    privateData: true,
  },
  {
    name: "CAD4TB v6 → v7",
    family: "Commercial",
    task: "Abnormality score (0–100)",
    study: "Field validation / cost-effectiveness",
    dataset: "Field-deployed screening cohorts (e.g. Pakistan)",
    metric: "AUC 0.823 (v6) → 0.903 (v7) · 132 subjects/day at $5.95/screen",
    value: 90.3,
  },
  {
    name: "qXR (Qure.ai) & Lunit INSIGHT CXR",
    family: "Commercial",
    task: "Abnormality score + finding localisation",
    study: "Large-scale screening cohorts (e.g. Brazil prisons)",
    dataset: "Real-world screening populations",
    metric: "spec > 70% at fixed 90% sens — meets WHO TPP",
    value: 70,
  },
];

export interface DatasetRow {
  name: string;
  size: string;
  composition: string;
  tasks: string;
  role: string;
}

export const FOUNDATION_DATASETS: DatasetRow[] = [
  {
    name: "JSRT",
    size: "247",
    composition: "154 nodule, 93 normal; 12-bit, 2048×2048",
    tasks: "Detection, segmentation",
    role: "Nodule / segmentation benchmark",
  },
  {
    name: "Montgomery County",
    size: "138",
    composition: "58 TB+, 80 normal; includes lung masks",
    tasks: "Segmentation, classification",
    role: "U-Net / segmentation training",
  },
  {
    name: "Shenzhen",
    size: "662",
    composition: "336 TB+, 326 normal",
    tasks: "Classification",
    role: "Classification, transfer learning",
  },
  {
    name: "TBX11K",
    size: "11,200",
    composition: "5 categories with bounding boxes",
    tasks: "Object detection",
    role: "Detection (YOLO, Faster R-CNN) — this project",
  },
  {
    name: "MIMIC-CXR / ChestX-ray14",
    size: "100,000+",
    composition: "Up to 14 thoracic pathologies, not TB-specific",
    tasks: "Multi-label classification",
    role: "Pretraining before TB fine-tuning",
  },
];
