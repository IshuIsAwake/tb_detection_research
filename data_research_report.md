# TB Dataset Research Report

---

## Quick Reference Table

| Dataset | Link | TB Images | Annotation Type | BBox | Seg Mask | Human / AI | Access | Format | Use Cases | Short Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **TBX11K** | [GitHub](https://github.com/yun-liu/Tuberculosis) | ~800 train+val (1,211 boxes) | Image-level class (6 categories) + BBox | ✅ Active + Latent | ❌ | ✅ Human — 2 radiologists, double-checked | Free | PNG 512×512 | Detection, classification, YOLO training | Only dataset with bbox + subtype labels. Test GT withheld for competition. |
| **Shenzhen Hospital CXR** | [NLM](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html) | 336 | Image-level (TB/Normal) + pixel masks per lesion type | ❌ | ✅ 19 lesion types | ✅ Human — 2 radiologists (junior + senior) | Free | PNG (original res) | TB lesion segmentation, classification | Masks added retroactively by separate research team. First public pixel-level TB annotation. |
| **Montgomery County CXR** | [NLM](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html) | 58 | Image-level (TB/Normal) + lung segmentation masks | ❌ | ✅ Lung only (not TB lesions) | ✅ Human — NLM radiologists | Free | PNG ~4000×4900 | Lung segmentation, classification baseline, external test set | Gold standard for lung field segmentation. Very small. Used as OOD test set in most TB papers. |
| **NIAID TB Portals** | [tbportals.niaid.nih.gov](https://tbportals.niaid.nih.gov/access-data) | 5,000+ CXR + CT | Image-level class + finding-level manual + Qure.ai AI annotations | ⚠️ Partial (~22% manually annotated) | ❌ (official) | ✅ Human + AI (kept separate) + genomic + clinical records | DUA required | DICOM | Multimodal research, drug-resistance prediction, classification, segmentation (via NIAID repo) | Most data-rich source. Has CTs, genomics, drug resistance profiles. Majority of CXRs lack spatial annotation. Focuses on MDR/XDR-TB. |
| **NIAID TB Lesion Segmentation Repo** | [GitHub](https://github.com/niaid/tb_lesion_cxr_segmentation) | ~6,328 (from TB Portals) | Pixel-level segmentation masks — "Secondary Pulmonary Tuberculosis" | ❌ | ✅ TB lesion masks | ⚠️ Single annotator ("Zhying") | Pretrained weights free; data via TB Portals DUA | PNG / DICOM | TB lesion segmentation (UNet, YOLOv8-seg, nnUNet) | Largest TB segmentation dataset by far. Single annotator is a limitation. Pretrained weights usable immediately without DUA. SPIE 2025 paper. |
| **NIH ChestX-ray14** | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | ❌ No TB label | Image-level (14 thoracic diseases) | ✅ ~880 images only | ❌ | ❌ NLP-extracted from radiology reports (NegBio) | Free | PNG 1024×1024 | Pretraining / transfer learning backbone | Not a TB dataset. Labels are AI-generated with ~5–10% noise. Known train/val patient leakage. Value is scale (112,120 images) for pretraining. |
| **VinDr-CXR** | [Kaggle PNG](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) / [PhysioNet DICOM](https://physionet.org/content/vindr-cxr/1.0.0/) | ❌ No TB label | Image-level class + BBox (14 findings) | ✅ All 18,000 images | ❌ | ✅ Human — 3 rads per train image, 5-consensus for test | Kaggle PNG: free; DICOM: PhysioNet click-through DUA | PNG 1024×1024 / DICOM | Pretraining / transfer learning, finding-level detection | Highest annotation quality of any public CXR dataset. No TB label but TB manifestations (consolidation, nodule, etc.) present. Use for pretraining before fine-tuning on TBX11K. |
| **Rahman / Kaggle TB Database** | [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | 700 free / 3,500 full | Image-level only (TB / Normal) | ❌ | ❌ | ⚠️ Inherited from source datasets (NLM, NIAID, Belarus, RSNA) | Free (700 images); NIAID DUA for full 3,500 | PNG/JPG | OOD / cross-dataset evaluation, classification baseline | Compiled from multiple sources — not original. Only 56 images actually from NIAID. Adds little beyond what other datasets already cover. Useful for OOD testing only. |

---

## Datasets — Detailed Notes

---

### 1. TBX11K

**Source & Access**

- Original dataset from Nankai University and InferVision, published at CVPR 2020.
- Official page and download: [https://github.com/yun-liu/Tuberculosis](https://github.com/yun-liu/Tuberculosis)
- License: CC BY 4.0. Freely available, no agreement required.
- Also browsable with statistics on Dataset Ninja: [https://datasetninja.com/tbx-11k](https://datasetninja.com/tbx-11k)

**What It Contains**

- 11,200 chest X-ray images at 512×512 pixels (downsampled from original ~3000×3000 clinical resolution).
- Every image has an image-level classification label in one of 6 categories: `healthy`, `sick_but_non_tb`, `active_tb`, `latent_tb`, `active_tb & latent_tb`, `uncertain_tb`.
- TB-positive images additionally have bounding box annotations with two box-level classes: `ActiveTuberculosis` and `ObsoletePulmonaryTuberculosis`.
- Total bounding boxes: 1,211 across ~799 TB-positive train+val images.
- Average ~1.5 boxes per TB image. Box sizes range from 0.23% to 26.66% of image area.

**Splits**

| Split | Images | Notes |
|---|---|---|
| Train | 6,600 | Full labels + boxes available |
| Val | 1,800 | Full labels + boxes available |
| Test | 3,302 | Labels withheld — online competition server only |

**Class Distribution**

| Category | Total | Train | Val | Test |
|---|---|---|---|---|
| Healthy | 5,000 | 3,000 | 800 | 1,200 |
| Sick non-TB | 5,000 | 3,000 | 800 | 1,200 |
| Active TB | 924 | 473 | 157 | 294 |
| Latent TB | 212 | 104 | 36 | 72 |
| Active + Latent | 54 | 23 | 7 | 24 |
| Uncertain TB | 10 | 0 | 0 | 10 |

**Annotation Quality**

- All image-level labels confirmed against gold standard (diagnostic microbiology).
- Bounding boxes drawn by a radiologist with 5–10 years TB experience, then verified by a second radiologist with 10+ years experience.
- Double-check protocol: mismatches sent back for re-annotation blind; if wrong twice, annotators discuss with gold standard reference.
- No AI involvement in annotation.

**YOLO Format Mapping**

- Healthy and sick-non-TB images → empty label files (background negatives, ~10,000 of them).
- Active TB images → class 0 boxes (`ActiveTuberculosis`).
- Latent TB images → class 1 boxes (`ObsoletePulmonaryTuberculosis`).
- Active+Latent images → mix of class 0 and class 1 boxes in one label file.
- Uncertain TB images → in test set only, no boxes released.

**Key Nuances**

- The only public TB dataset combining image-level subtype classification with bounding box localization.
- ~93% of images have empty label files (no boxes) — this is correct, not missing data.
- Test set ground truth deliberately withheld; use train+val for all local experiments.
- 512×512 resolution chosen deliberately for CNN compatibility; full 3000×3000 DICOMs not released.
- The paper notes experienced radiologists only achieved 68.7% accuracy vs gold standard, contextualising why the task is hard.

---

### 2. Shenzhen Hospital CXR Set

**Source & Access**

- Collected from Shenzhen No. 3 People's Hospital, China. Released by the US National Library of Medicine (NLM).
- Direct download: [https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html)
- Free, no agreement required. Originally published 2014. Pixel masks added later by a separate research team (published 2022 in MDPI Data journal).

**What It Contains**

The folder structure has four components:

`CXR_png/` — 662 chest X-ray images in PNG format. Filename convention: `CHNCXR_XXXX_Y.png` where `Y=0` is normal, `Y=1` is TB.

`ClinicalReadings/` — one `.txt` file per image. Very short (15–23 bytes) — contains a brief radiologist note such as "normal" or "tuberculosis, upper lobe infiltrate". These are the original image-level labels.

`Annotations/masks/` — pixel-level segmentation masks for TB-positive images only. Each mask is a separate PNG named after the image and the finding type, e.g.:
- `CHNCXR_0330_1_Cavity_1.png`
- `CHNCXR_0348_1_Severe_Infiltrate_(Consolidation)_4.png`
- `CHNCXR_0360_1_Clustered_Nodule_(2mm-5mm_apart)_3.png`

`Annotations-2/` — a second independent set of masks from a different annotator, for inter-rater comparison.

`shenzhen_consensus_roi.csv` — consensus ROI file reconciling both annotation sets.

**The 19 Lesion Types in the Masks**

Cavity, Calcified Nodule, Single Nodule (non-calcified), Clustered Nodule, Linear Density, Small Infiltrate, Moderate Infiltrate, Severe Infiltrate (Consolidation), Pleural Effusion, Pleural Thickening (apical), Pleural Thickening (non-apical), Adenopathy, Apical Thickening, Retraction, Thickening of interlobar fissure, Calcification (other), Unknown, Other.

**Annotation Quality**

- Original image-level labels: clinical diagnosis from Shenzhen hospital records.
- Pixel masks (added 2022): annotated by two radiologists from the Chinese University of Hong Kong using the Firefly annotation tool. Junior radiologist labeled first, senior radiologist verified all, consensus reached on disagreements.
- First published pixel-level annotation of TB findings in CXRs.

**Key Nuances**

- Normal images (326 total) have clinical reading text only — no masks, because there is nothing to segment.
- Only TB-positive images (336) have masks; of those, 330 actually had observable TB signs (6 were labeled TB clinically but radiologists found no visible findings to mark).
- Masks are per-lesion-instance PNG files, not a single combined mask per image — one image can have 10+ separate mask files.
- Binary classification only at image level — no Active/Latent distinction unlike TBX11K.
- The `Annotations-2/` folder uses the same structure but comes from a different annotator team, useful for inter-rater agreement analysis.
- Commonly used as an external generalization test set in papers that train on TBX11K.

---

### 3. Montgomery County CXR Set

**Source & Access**

- Collected from the tuberculosis screening program of Montgomery County Department of Health and Human Services, Maryland, USA. Collaboration between Montgomery County and the NLM/NIH.
- Direct download: [https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html)
- Free, no agreement required. Published 2014. IRB exempted by NIH (No. 5357).
- Cite: Jaeger et al., IEEE Trans Med Imaging, 2014. DOI: 10.1109/TMI.2013.2290491

**What It Contains**

`CXR_png/` — 138 posterior-anterior chest X-rays. Filename convention: `MCUCXR_XXXX_Y.png` where `Y=0` is normal, `Y=1` is TB.

`ClinicalReadings/` — one `.txt` per image, ~45 bytes each. Contains patient age, gender, and lung status — notably more structured than Shenzhen's one-word notes.

`ManualMask/leftMask/` and `ManualMask/rightMask/` — manual pixel-level segmentation masks of the left and right lung fields separately. All 138 images have both masks — normal and TB cases alike.

`montgomery_consensus_roi.csv` — consensus ROI file.

**Composition**

- 80 normal images
- 58 TB-positive images
- Covers a wide range of TB manifestations including effusions and miliary patterns.
- Original image resolution: 4020×4892 or 4892×4020 pixels. Pixel spacing 0.0875mm. 12-bit grayscale — proper clinical-grade radiographs.

**Annotation Quality**

- Image-level labels from the tuberculosis screening program's clinical records.
- Lung masks: manually traced by NLM researchers. Every single image annotated regardless of TB status.
- No AI involvement.

**Key Nuances**

- The masks are of the **lung fields**, not TB lesions. This dataset is primarily a lung segmentation benchmark, not a TB localization dataset.
- Despite being only 138 images, it is one of the most cited TB datasets in existence — largely because it was one of the first properly de-identified public TB CXR datasets from a government institution.
- Binary classification at image level only — no subtypes.
- The high original resolution (4000+ pixels) makes it suitable for fine-grained lung analysis if you work from the original PNGs rather than downsampled versions.
- Standard use in TB research: train on TBX11K, evaluate generalization on Montgomery County and Shenzhen as external OOD test sets.

---

### 4. NIAID TB Portals

**Source & Access**

- Multi-national collaboration coordinated by NIAID (National Institute of Allergy and Infectious Diseases), NIH.
- Data request portal: [https://tbportals.niaid.nih.gov/access-data](https://tbportals.niaid.nih.gov/access-data)
- Two separate DUA processes:
  - **Clinical data** (patient metadata, lab results, treatment, outcomes): via [accessclinicaldata.niaid.nih.gov](https://accessclinicaldata.niaid.nih.gov)
  - **Imaging + genomics** (CXR/CT DICOM files, pathogen genomic sequences): via the TB Portals form directly at the link above.
- If you need all three (clinical + imaging + genomics), you must submit two separate requests.
- Access methods after approval: Aspera fileshare download or programmatic Data API.

**What It Contains**

The imaging download includes:

- **CXR images** (DICOM format) — from multiple countries, tied to patient cases via `patient_id`. Includes both digitized film X-rays (camera or flatbed scanner) and direct digital. Both grayscale and color images present (color = camera-digitized film). HDR images (0–65535) and LDR images (0–255) mixed.
- **CXR Manual Annotations** — finding-level annotations added by a single radiologist per image. Includes presence of cavitation, Affected Lung Percentage (ALP), and other findings that enable computation of the Timika score. As of an earlier snapshot: only ~22% of CXRs manually annotated by a radiologist, ~6.7% by a pulmonologist, remaining ~70% unannotated spatially.
- **CXR Qure.ai Annotations** — AI-generated annotations from Qure.ai's qXR tool, covering cavity, nodule, pleural effusion, hilar lymphadenopathy. Kept as a separate file from manual annotations.
- **CT images** (DICOM series) — 3D volumetric scans per patient.
- **CT Annotations** — annotation reports per CT study.
- **Genomic data** — pathogen genomic sequences, SRA metadata, gene variants.
- **Drug Susceptibility Testing (DST)** — results for first- and second-line drugs including microscopy, culture, MGIT Bactec, GeneXpert.

**Patient Coverage**

- 3,400+ international TB patient cases from multiple countries in Eastern Europe, Asia, and sub-Saharan Africa.
- Heavily focused on **drug-resistant TB** — MDR-TB (multi-drug resistant) and XDR-TB (extensively drug resistant). This is clinically the most urgent TB subset and the one most underrepresented in other datasets.
- ~5,000 CXRs as of 2023 releases (growing over time as it is an ongoing collection effort).

**Annotation Quality**

- Image-level labels: lab-confirmed clinical diagnosis — gold standard.
- Manual spatial annotations: single radiologist per image (finding-level, not pixel masks).
- Qure.ai annotations: commercial AI tool, provided separately and clearly labelled as AI-generated.
- No double-check protocol on manual annotations unlike TBX11K.

**Key Nuances**

- The only dataset linking imaging to genomics and drug resistance profiles per patient — enables multimodal research.
- CT scans are true 3D volumes — directly compatible with nnUNet in 3D mode (unlike CXRs which are 2D).
- The annotation CSV referenced in the NIAID segmentation repo (`TB_Portals_labeled20231121.csv`) is a specific annotation file from a researcher ("Zhying") covering ~6,328 CXRs — this is obtained as part of the imaging data request.
- Images are heterogeneous: mixed resolutions, mixed HDR/LDR, mixed grayscale/color. Preprocessing pipeline is non-trivial and documented in the NIAID segmentation repo.

---

### 5. NIAID TB Lesion Segmentation Repository

**Source & Access**

- Published by NIAID's Bioinformatics and Computational Biosciences Branch (BCBB).
- GitHub: [https://github.com/niaid/tb_lesion_cxr_segmentation](https://github.com/niaid/tb_lesion_cxr_segmentation)
- License: Apache 2.0. Code and pretrained weights are freely available with no agreement required.
- Pretrained weights are stored via git-lfs — requires `git lfs install` before cloning.
- Underlying data (TB Portals CXRs + annotation CSV) requires TB Portals imaging DUA (same request as above).
- Reference paper: Kantipudi et al., SPIE Medical Imaging 2025. DOI: [10.1117/12.3047222](https://doi.org/10.1117/12.3047222)

**What It Contains**

Three parallel segmentation pipelines all trained on the same data:

**UNet-ResNet18** — encoder-decoder with ResNet18 backbone initialized on ImageNet weights. Pretrained weight: `segment_tb_cxr/unet_resnet18/weights/customunet.pt`

**YOLOv8 (instance segmentation)** — YOLOv8m-seg initialized on COCO weights, fine-tuned for TB lesion mask prediction. Pretrained weight: `segment_tb_cxr/yolov8/weights/yolov8.pt`

**nnUNet** — automated segmentation framework with self-configuring pipeline. Pretrained weight: `segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth` (with accompanying `plans.json` and `dataset.json`).

**Ensemble mode** — combines YOLOv8 and nnUNet predictions, output in `.nrrd` format.

**Classification module** — `classification_tb_not_tb/` directory handles binary TB/not-TB classification as a separate task.

**Hyperparameter optimization** — Optuna-based pipeline with PostgreSQL backend for parallelized search, designed for SLURM cluster environments.

**Docker support** — Dockerfile included. Output from Docker inference: per-image CSV with columns `Image`, `TB Lesion Contours`, `TB Score`, `Prediction`.

**Dataset Used for Training**

- Source: TB Portals CXRs (August 2023 snapshot) annotated with `TB_Portals_labeled20231121.csv`.
- Abnormality class: `"Secondary Pulmonary Tuberculosis"` — a single unified label for all TB lesion pixels.
- Approximate split: Train 4,429 / Val 949 / Test 950 (total ~6,328 images).
- Single annotator — "Zhying" — no double-check protocol documented.

**Key Nuances**

- Largest TB lesion segmentation dataset and pipeline currently available publicly.
- Pretrained weights can be used for immediate inference on any CXR — no DUA needed for inference alone.
- YOLOv8 is used here in **segmentation mode** (instance segmentation masks), not detection mode — different from standard YOLO bbox usage.
- nnUNet here operates in **2D mode** despite being typically used for 3D volumetric data — CXRs are 2D. If you use this with CT data from the portal, switch to 3D nnUNet configuration.
- The annotation label `"Secondary Pulmonary Tuberculosis"` collapses all lesion types (cavities, nodules, infiltrates etc.) into one binary mask — no lesion-type distinction unlike Shenzhen's 19 classes.
- Note: repo has 0 stars and 1 fork as of the research date — very new and low visibility despite being from NIAID directly.

---

### 6. NIH ChestX-ray14

**Source & Access**

- Published by NIH Clinical Center, 2017. Wang et al., CVPR 2017.
- Kaggle mirror (most commonly used): [https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- Original source: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- Free, no agreement required. Supported by NIH Intramural Research Program.

**What It Contains**

- 112,120 frontal-view chest X-rays (PA and AP) from 30,805 unique patients at the NIH Clinical Center.
- Each image has image-level multi-label annotations for 14 thoracic diseases: Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia.
- Additionally: a "No Finding" label (~54% of images, 60,361 cases).
- ~880 images have radiologist-drawn bounding boxes across 8 of the 14 pathologies — intended for localization evaluation only.
- Patient metadata per image: patient ID, age, gender, view position (PA or AP).
- Images standardized to 1024×1024 pixels.

**Annotation Method — NLP Pipeline**

Labels were not annotated by radiologists looking at images. The pipeline was:
1. Radiologists wrote free-text radiology reports (standard clinical workflow).
2. NLP tool (NegBio) mined those reports for disease mentions, negations, and uncertainty.
3. Image-level labels extracted automatically from text output.

Original reports are not publicly released. Estimated label accuracy: ~90–95%. Estimated noise: 5–10% based on independent studies.

**Known Issues**

- **No TB label** — tuberculosis is not among the 14 classes.
- **Label noise** — NLP extraction errors, particularly for rare findings and negation handling.
- **Train/val patient leakage** — 67.4% of the validation set shares patients with the training set in the official splits. Any published results using these official splits are inflated.
- **Severe class imbalance** — "No Finding" is 54% of the dataset; Hernia is 0.2%.
- No large-scale manual re-labeling has been released to correct these issues.

**Key Nuances**

- Relevance to this project is for **pretraining only** — the scale (112,120 images) makes it valuable for learning general chest X-ray features before fine-tuning on TB-specific data.
- The CheXNet model (DenseNet-121 trained on this dataset) is a very common pretrained backbone used in TB detection literature. Its claimed "radiologist-level" performance was heavily criticised due to the patient leakage problem.
- Always use patient-level splits (all images from one patient go to one split only) when using this dataset to avoid leakage.

---

### 7. VinDr-CXR (VinBigData)

**Source & Access**

- Created by Vingroup Big Data Institute (VinBigdata), Vietnam. Collected from two major Vietnamese hospitals.
- Published as Scientific Data paper (Nature), 2022: [https://www.nature.com/articles/s41597-022-01498-w](https://www.nature.com/articles/s41597-022-01498-w)
- Two access points:
  - **Kaggle** (PNG 1024×1024, free, no agreement): [https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
  - **PhysioNet** (DICOM, click-through DUA): [https://physionet.org/content/vindr-cxr/1.0.0/](https://physionet.org/content/vindr-cxr/1.0.0/)
- The Kaggle version was used for the competition and has a slightly modified version of the data. For the full released dataset use PhysioNet.

**What It Contains**

- 18,000 frontal chest X-ray images total (15,000 train, 3,000 test).
- Source pool: 100,000+ scans from two hospitals; 18,000 selected and annotated.
- Adult patients only.

Each training image annotation row in `annotations_train.csv` contains: `image_id`, `rad_id`, `class_name`, `x_min`, `y_min`, `x_max`, `y_max`.

**The 14 Finding Classes + No Finding**

Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation, ILD (Interstitial Lung Disease), Infiltration, Lung Opacity, Nodule/Mass, Other lesion, Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis, No finding.

**Annotation Quality — The Key Differentiator**

- **Training set**: each image independently labeled by 3 radiologists.
- **Test set**: each image annotated by consensus of 5 radiologists.
- 17 radiologists total involved across the full dataset.
- Annotation performed via VinDr Lab, a custom DICOM annotation platform built on top of a PACS system.
- No AI involvement. Fully human annotation throughout.
- This is the highest annotation quality of any public chest X-ray dataset currently available.

**Multi-Annotator Structure**

Because training images have 3 independent annotations, the raw CSV has multiple bounding box rows per image (one per radiologist per finding). Before using in YOLO training you must merge/reconcile these. Standard approach: Weighted Box Fusion (WBF) to create consensus boxes.

**Key Nuances**

- **No TB label** — findings-based like ChestX-ray14, but TB manifestations (consolidation, calcification, nodule, infiltration, pleural effusion) are all present and labeled.
- Relevance to this project: pretraining and transfer learning. A model pretrained on VinDr-CXR's human-verified bounding boxes then fine-tuned on TBX11K is a significantly stronger starting point than one pretrained on ChestX-ray14's noisy NLP labels.
- The Kaggle PNG version (1024×1024) is sufficient for YOLO training — DICOM provides metadata but PNG is fine for image-level work.
- Raw annotation CSV for training contains ~211,000 rows (multiple radiologists × multiple findings × 15,000 images) — expect to preprocess before training.

---

### 8. Rahman / Kaggle TB Chest X-ray Database

**Source & Access**

- Compiled by researchers from Qatar University, University of Dhaka, and collaborators from Malaysia and Hamad Medical Corporation, Bangladesh.
- Kaggle: [https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- IEEE DataPort (subscription required for full metadata): DOI: 10.1109/ACCESS.2020.3031384
- Free on Kaggle for the 700-image version. Full 3,500 TB images require NIAID TB Portals DUA.

**What It Contains**

- Two folders: `TB/` and `Normal/`. Folder name is the label. That is the entirety of the annotation.
- Current release: 3,500 TB images + 3,500 Normal images = 7,000 total.
- No bounding boxes, no segmentation masks, no patient metadata, no lesion types.

**Actual Source Breakdown**

The dataset is a compiled aggregate — not original data collection:

TB images sourced from:
- NIAID TB Portals (~2,800 images, requires their DUA)
- Belarus TB Portal program (30 images)
- NLM datasets — Montgomery County + Shenzhen (14 images)
- Qure.ai / NIAID annotations (56 images)

Normal images sourced from:
- NLM datasets (40 images)
- RSNA Pneumonia Detection Challenge dataset (60 images)

**Annotation Quality**

Labels are inherited from source datasets — no new annotation was performed. Quality depends entirely on what the source datasets used. Provenance for the bulk NIAID portion is the least transparent part.

**Key Nuances**

- Calling this the "NIAID TB dataset" (common in literature) is misleading — only ~56 images directly from NIAID; the bulk is from the NIAID portal via a separate agreement the authors held.
- Adds no annotation type not already present in the constituent datasets.
- Best use case: cross-dataset / OOD evaluation. Since it aggregates from multiple sources and hospitals, a model that performs well on it has demonstrated some generalization.
- Do not use for training if you already have TBX11K — the classification labels here are a strict subset of what TBX11K provides, at lower annotation quality and without any spatial annotation.

---

*Document compiled during initial dataset research phase. No specific modelling goal assumed.*
