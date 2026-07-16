# Deep Learning Models for TB Detection: Datasets & Performance Metrics

This table compiles the specific dataset(s) and reported performance metrics for each architecture, as described in *Exhaustive Analysis of Deep Learning Architectures for Tuberculosis Detection in Chest Radiography*.

| Architecture | Task Type | Study / Source | Dataset(s) Used | Reported Metrics |
|---|---|---|---|---|
| **AlexNet + GoogLeNet (ensemble)** | Classification (binary: TB vs. normal) | Lakhani & Sundaram (2017), *Radiology* | Private dataset of 1,007 posteroanterior CXRs | AUC 0.99; accuracy 95.3%; sensitivity 92.0%; specificity 98.7%. With radiologist tiebreaker on 13 disagreements: sensitivity 97.3%, specificity 100% |
| **AlexNet (pretrained)** | Classification (binary) | Transfer-learning study on microscopic slide images | Microscopic slide images (not standard CXR) | Accuracy 98.73%; sensitivity 98.59% |
| **VGG-16** | Classification (binary) | Comparative CNN benchmarking study (Huang et al., 2025) | Private dataset, 4,200 CXRs (700 TB-positive) | Accuracy 99.4%; precision 97.9%; recall 98.6%; F1 98.3%; AUC-ROC 98.25% |
| **ResNet (multi-class variants)** | Classification (multi-class: normal / viral pneumonia / bacterial pneumonia / TB) | Multi-class hierarchical classification studies | Not specified (multi-class CXR cohort) | Accuracy >96% |
| **ResNet (TBNet, semi-supervised)** | Classification (binary, semi-supervised) | Liu et al. | >100,000 unlabeled radiographs (semi-supervised) + external test set | AUC 0.91 (external test set) |
| **DenseNet-169 + SVM/DNN classifier** | Classification (binary) | Ganapathy et al. | Evaluated across **three public datasets** (unspecified which three, likely JSRT/Montgomery/Shenzhen) | Accuracy 99.91%; precision 99.23%; recall 99.22%; AUC ~99.99% |
| **MobileNet + Artificial Ecosystem Optimization (AEO)** | Classification (binary, with feature selection) | Sahlol et al. (2020) | External validation dataset (feature set reduced from 50,000 → <25 features) | Accuracy 94.1% (external validation) |
| **EfficientNetV2 (multimodal ensemble)** | Classification (binary/multimodal fusion) | Multimodal clinical decision support integrations | CXR + structured patient data (multimodal) | Accuracy >90% |
| **U-Net** | Segmentation (lung field isolation) | Sharma et al. (2023) | Montgomery + Shenzhen (standard segmentation benchmarks) | Accuracy 96.35%; Jaccard index 90.38%; Dice coefficient 94.88% |
| **Attention U-Net + Swin Transformer (combined pipeline)** | Segmentation (lung masking) + Classification (downstream) | Visu et al. (2025) | Not specified (CXR segmentation + classification pipeline) | Overall diagnostic accuracy >99% |
| **Faster R-CNN** | Object detection (lesion localization) | Xie et al. (2020) | Shenzhen + Montgomery | AUC 0.977; accuracy 92.6% |
| **YOLO (general, real-time variants)** | Object detection (real-time lesion bounding boxes) | Various contemporary implementations | Not specified | ~68% mean Average Precision (mAP); optimized for real-time speed over precision |
| **YOLO-CXR (with Selective Feature Fusion)** | Object detection (small/early lesion localization) | Specialized variant for small/early lesions | Not specified | Improved detection rate for small, early-stage TB lesions (no single headline metric given) |
| **ViT + CLAHE preprocessing (hybrid)** | Classification (binary) | Hybrid transformer study | Not specified | Average precision 99.90%; accuracy up to 99.84%; F1 99.71% |
| **Enhanced Swin Transformer (EnSTrans) + Residual Pyramid Network** | Classification (binary) + Segmentation (pyramid network) | Visu et al. (2025) | Not specified | Accuracy 99.05% |
| **TB-Net (self-attention CNN)** | Classification (binary) | Wong et al. (2021) | Multi-national screening cohorts | Accuracy 99.86%; sensitivity 100% |
| **Vision Mamba CGSM (Context-Guided Slot Mixing)** | Classification (multi-class + fine-grained binary: active vs. inactive TB) | Jeon et al. (2026) | Hierarchical cohort: Normal / Pneumonia / Active TB / Inactive TB | Multi-class accuracy 92.96%. Active vs. Inactive TB (binary): specificity 97.04%, Youden Index 79.55% — outperforming ResNet-152 and ViT-B baselines |
| **Vision Mamba (general)** | Classification (binary, efficiency-focused) | Comparative Mamba vs. CNN/ViT study | Not specified | ~80% reduction in GPU memory vs. standard ViT, at comparable diagnostic accuracy |
| **RepViT-CXR (CNN-ViT hybrid)** | Classification (binary, TB and pneumonia) | arXiv preprint | Multiple datasets (unspecified) | New state-of-the-art benchmarks for TB and pneumonia diagnosis (specific numbers not given) |
| **GhostNet + MobileViT (lightweight hybrid)** | Classification (binary, efficiency-focused) | Owda et al. | Specific dataset not named | Accuracy >99%; 7.73M parameters; 282.11M FLOPs |
| **CAD4TB v6 → v7 (commercial, deep-learning-based)** | Classification (continuous abnormality score, 0–100) | Field validation / cost-effectiveness studies | Field-deployed screening cohorts (e.g., Pakistan) | AUC improved from 0.823 (v6) to 0.903 (v7); v6 cost-effectiveness: 132 subjects/day at $5.95/screen at 90% sensitivity threshold |
| **qXR (Qure.ai) & Lunit INSIGHT CXR (commercial)** | Classification (abnormality scoring) + Object detection (discrete finding localization) | Large-scale screening cohort studies (e.g., incarcerated populations in Brazil) | Real-world screening populations | Specificity >70% at fixed 90% sensitivity (meets WHO Target Product Profile) |

## Reference: Foundational Public Datasets

| Dataset | Size | Composition | Task Type(s) Supported | Primary Role |
|---|---|---|---|---|
| **JSRT** | 247 images | 154 nodule cases, 93 normal; 12-bit grayscale, 2048×2048 | Object detection (nodules), segmentation | Nodule detection, segmentation benchmark |
| **Montgomery County (MC)** | 138 images | 58 TB-positive, 80 normal; includes lung segmentation masks | Segmentation, classification | U-Net/segmentation training |
| **Shenzhen** | 662 images | 336 TB-positive, 326 normal | Classification | General classification, transfer learning |
| **TBX11K** | 11,200 images | 5 categories with bounding boxes (healthy, sick non-TB, active TB, latent TB, uncertain) | Object detection | Object detection (YOLO, Faster R-CNN) |
| **MIMIC-CXR / ChestX-ray14** | 100,000+ images | Up to 14 thoracic pathologies (not TB-specific) | Classification (multi-label), pretraining | Pretraining before TB fine-tuning |

## Notes on Data Gaps
- Several high-performing results (VGG16's 99.4%, DenseNet-169's 99.91%, ViT hybrids near 99.9%) come from **studies using private or unspecified datasets**, not always the standard JSRT/Montgomery/Shenzhen trio — worth keeping in mind when comparing numbers across rows, since dataset difficulty and size vary considerably.
- Some entries (YOLO general, RepViT-CXR, Vision Mamba general) don't have a single named dataset or headline metric in the source document — these are noted as "not specified."
- Where a study says "evaluated across three public datasets" without naming them, this is left as stated in the source rather than assumed.
