# Literature review — reading list

Foundational papers for the architecture decisions in this project (which
backbone, which detector, how to pretrain). This is a **reference list, not the
review itself** — titles, links, and a one-line note on why each matters here.

The "Done" column is just a personal tracker — tick papers off as you read.

## Reading order

The papers build on each other; read them in dependency order, not by date:

1. **Backbones first** — ResNet → FPN. They sit under everything downstream.
2. **Then the two detector chains in parallel** — this is the actual
   YOLO vs RetinaNet vs Faster R-CNN decision:
   - two-stage: R-CNN → Fast R-CNN → Faster R-CNN
   - one-stage: YOLOv1 → SSD → RetinaNet
3. **Ground it in the domain** — TBX11K (the dataset) + CheXNet + MoCo-CXR.
4. **Pretraining** — BYOL / DINO, when you return to the Kaggle SSL decision.
5. **Next phase (after CNNs)** — Attention Is All You Need → ViT → Swin, then
   the SSM line (S4 → Mamba → Vision Mamba).

---

## 1. Backbones / classification

Read first — the "spine" for both the classifier and the detector.

| Done | Paper | Authors · venue | Link | Why it matters here |
|---|---|---|---|---|
| [ ] | **ResNet — Deep Residual Learning for Image Recognition** | He et al. · CVPR 2016 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | The skip-connection paper. The backbone candidate for both stages. |
| [ ] | VGG — Very Deep Convolutional Networks | Simonyan & Zisserman · ICLR 2015 | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) | The "stack 3×3 convs" predecessor; context for why ResNet mattered. |
| [ ] | Batch Normalization | Ioffe & Szegedy · ICML 2015 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) | Enabling trick used everywhere after. |
| [ ] | EfficientNet | Tan & Le · ICML 2019 | [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) | Compound scaling; a classifier-backbone alternative. |
| [ ] | ConvNeXt — A ConvNet for the 2020s | Liu et al. · CVPR 2022 | [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | Modernized ResNet; strong current CNN backbone. |

## 2. Two-stage detectors (R-CNN lineage)

Read in order — each fixes the previous one's bottleneck.

| Done | Paper | Authors · venue | Link | Why it matters here |
|---|---|---|---|---|
| [ ] | R-CNN — Rich feature hierarchies | Girshick et al. · CVPR 2014 | [arXiv:1311.2524](https://arxiv.org/abs/1311.2524) | Where region-based detection starts. |
| [ ] | Fast R-CNN | Girshick · ICCV 2015 | [arXiv:1504.08083](https://arxiv.org/abs/1504.08083) | ROI pooling; the speedup step. |
| [ ] | **Faster R-CNN** | Ren et al. · NeurIPS 2015 | [arXiv:1506.01497](https://arxiv.org/abs/1506.01497) | Introduces the RPN — the candidate two-stage detector. Classically strong small-lesion recall. |
| [ ] | **FPN — Feature Pyramid Networks** | Lin et al. · CVPR 2017 | [arXiv:1612.03144](https://arxiv.org/abs/1612.03144) | **The unsung critical one for us** — multi-scale features, the main lever for small-lesion recall. Used by RetinaNet and nearly every modern detector. Don't skip. |
| [ ] | Mask R-CNN | He et al. · ICCV 2017 | [arXiv:1703.06870](https://arxiv.org/abs/1703.06870) | Adds segmentation; relevant only if we ever use masks. |

## 3. One-stage detectors

| Done | Paper | Authors · venue | Link | Why it matters here |
|---|---|---|---|---|
| [ ] | **YOLOv1 — You Only Look Once** | Redmon et al. · CVPR 2016 | [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) | The original one-stage idea. |
| [ ] | YOLOv2 / YOLO9000 | Redmon & Farhadi · CVPR 2017 | [arXiv:1612.08242](https://arxiv.org/abs/1612.08242) | Anchors, multi-scale training. |
| [ ] | YOLOv3 | Redmon & Farhadi · 2018 | [arXiv:1804.02767](https://arxiv.org/abs/1804.02767) | Widely-cited tech report; FPN-style multi-scale heads. |
| [ ] | YOLOv4 | Bochkovskiy et al. · 2020 | [arXiv:2004.10934](https://arxiv.org/abs/2004.10934) | Last YOLO with a thorough paper ("bag of freebies/specials"). |
| [ ] | YOLOv7 | Wang et al. · CVPR 2023 | [arXiv:2207.02696](https://arxiv.org/abs/2207.02696) | A peer-reviewed modern YOLO. |
| [ ] | **SSD — Single Shot MultiBox Detector** | Liu et al. · ECCV 2016 | [arXiv:1512.02325](https://arxiv.org/abs/1512.02325) | The other foundational one-stage; good contrast to YOLO. |
| [ ] | **RetinaNet / Focal Loss** | Lin et al. · ICCV 2017 | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | The candidate one-stage detector; focal loss targets *our* class imbalance directly. |

> **Citation note:** YOLOv5 and **YOLOv8 (the version this repo uses) have no
> peer-reviewed paper** — they're Ultralytics software releases. Cite v1–v4 / v7
> for the methodology and the [Ultralytics repo/docs](https://github.com/ultralytics/ultralytics)
> for v8 itself.

## 4. Anchor-free / transformer detectors (context, lower priority)

| Done | Paper | Authors · venue | Link | Why it matters here |
|---|---|---|---|---|
| [ ] | FCOS — Fully Convolutional One-Stage | Tian et al. · ICCV 2019 | [arXiv:1904.01355](https://arxiv.org/abs/1904.01355) | Anchor-free; YOLOv8 is anchor-free in this spirit. |
| [ ] | DETR — End-to-End Detection with Transformers | Carion et al. · ECCV 2020 | [arXiv:2005.12872](https://arxiv.org/abs/2005.12872) | Transformer detection. Data-hungry — likely wrong for 799 images, but worth knowing *why* we rule it out. |

## 5. Self-supervised pretraining (the Kaggle plan)

| Done | Paper | Authors · venue | Link | Why it matters here |
|---|---|---|---|---|
| [ ] | **BYOL — Bootstrap Your Own Latent** | Grill et al. · NeurIPS 2020 | [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) | The SSL method under discussion; no negatives, tolerant of small batch. |
| [ ] | SimCLR | Chen et al. · ICML 2020 | [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) | Contrastive baseline; needs large batches. |
| [ ] | MoCo | He et al. · CVPR 2020 | [arXiv:1911.05722](https://arxiv.org/abs/1911.05722) | Momentum contrast with a memory bank. |
| [ ] | DINO | Caron et al. · ICCV 2021 | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | Self-distillation; shines with ViT. |
| [ ] | MAE — Masked Autoencoders | He et al. · CVPR 2022 | [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) | Masked-image pretraining; ViT-centric. |
| [ ] | **MoCo-CXR** | Sowrirajan et al. · MIDL 2021 | [arXiv:2010.05352](https://arxiv.org/abs/2010.05352) | **SSL on chest X-rays specifically** — addresses the grayscale-augmentation problem. Read before committing to BYOL-on-CXR. |

## 6. Transformers & SSMs (next phase, after CNNs)

The line you're moving into. Attention paper first, then vision transformers,
then the state-space-model family.

| Done | Paper | Authors · venue | Link | Why it matters here |
|---|---|---|---|---|
| [ ] | **Attention Is All You Need** | Vaswani et al. · NeurIPS 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | The transformer. Foundation for everything below (currently reading). |
| [ ] | **ViT — An Image is Worth 16×16 Words** | Dosovitskiy et al. · ICLR 2021 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | Transformers as image backbones. |
| [ ] | Swin Transformer | Liu et al. · ICCV 2021 | [arXiv:2103.14030](https://arxiv.org/abs/2103.14030) | Hierarchical ViT; works as a detector backbone. |
| [ ] | S4 — Structured State Spaces | Gu et al. · ICLR 2022 | [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) | The modern SSM that started the line. |
| [ ] | **Mamba — Selective State Spaces** | Gu & Dao · 2023 | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Linear-time selective SSM; the one to know. |
| [ ] | Vision Mamba (Vim) | Zhu et al. · ICML 2024 | [arXiv:2401.09417](https://arxiv.org/abs/2401.09417) | Mamba as an image backbone. |
| [ ] | VMamba | Liu et al. · NeurIPS 2024 | [arXiv:2401.10166](https://arxiv.org/abs/2401.10166) | Alternative visual SSM backbone. |

## 7. Domain papers (CXR / TB) — must cite

| Done | Paper | Authors · venue | Link | Why it matters here |
|---|---|---|---|---|
| [ ] | **TBX11K — Rethinking Computer-aided Tuberculosis Diagnosis** | Liu et al. · CVPR 2020 | [project page](https://github.com/yun-liu/Tuberculosis) | **Our dataset paper — non-negotiable citation.** |
| [ ] | CheXNet | Rajpurkar et al. · 2017 | [arXiv:1711.05225](https://arxiv.org/abs/1711.05225) | DenseNet-121 on ChestX-ray14; canonical CXR-classification reference. |
| [ ] | ChestX-ray14 (NIH) | Wang et al. · CVPR 2017 | [arXiv:1705.02315](https://arxiv.org/abs/1705.02315) | Our NIH pretraining source. |
| [ ] | VinDr-CXR | Nguyen et al. · 2022 | [arXiv:2012.15029](https://arxiv.org/abs/2012.15029) | Our other pretraining source (bbox findings). |
