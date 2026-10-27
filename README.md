# DCENet: A Dual-Domain Collaborative Enhancement Network for Accurate and Lightweight Cattle Segmentation in Complex Farm Environments

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Task](https://img.shields.io/badge/Task-Cattle%20Segmentation-blueviolet)](#overview)

Official project repository for **DCENet**, a lightweight cattle segmentation network designed for challenging farm environments with complex backgrounds, changing illumination, occlusions, and varying animal poses.

**Repository:** https://github.com/LiuYiFei615/DCENet_cow

> **Release status:** The repository currently provides project and dataset information. Training/inference scripts, pretrained weights, and complete reproduction instructions are not yet available in the public repository. The README will be updated as these materials are released.

## Overview

DCENet adopts a U-shaped encoder–bottleneck–decoder architecture and combines complementary spatial-domain and frequency-domain feature enhancement. It aims to improve cattle body segmentation while maintaining a compact model size.

The two principal modules are:

- **Spatial-domain Topology-inspired Deformation Fusion Module (STDFM):** applied at a shallow stage to capture local spatial relationships through four-directional neighborhood differences and to adapt feature extraction to irregular shapes using deformable convolution. The module is *topology-inspired*; it does not claim mathematically guaranteed topological invariance.
- **Frequency-domain Phase Modulation Module (FPMM):** introduced at an intermediate stage to process spectral features. Its modulation phase function (MPF) adjusts phase information to enhance representations relevant to contours and fine structures.

The network also uses a lightweight **PCA–PSA bottleneck** with grouped operations to control computational complexity.

<!-- Add the verified network architecture illustration here when available:
![Architecture of DCENet](assets/DCENet_architecture.png)
-->

## Main Contributions

1. **Dual-domain collaboration:** combines spatial deformation-aware features with frequency-domain phase modulation at different encoder stages.
2. **Boundary-oriented feature learning:** introduces STDFM and FPMM to address irregular cattle contours, partial occlusions, and cluttered farm scenes.
3. **Lightweight design:** achieves competitive segmentation accuracy with approximately **0.45 million parameters** and **1.57 GFLOPs** per 256 × 256 input image.
4. **Evaluation on multiple sources:** assesses performance on a collected Jinnan cattle dataset and two publicly available cattle datasets, with separate training and validation on each source rather than zero-shot transfer.

## Datasets

### 1. Jinnan Cattle Segmentation Dataset

The collected **Jinnan Cattle Segmentation Dataset** includes **2,100 images** with manually annotated, pixel-level cattle segmentation masks. Images were collected in real farm environments with diverse lighting, backgrounds, poses, and occlusions.

- **Images and masks:** [Download from Quark Netdisk](https://pan.quark.cn/s/821cacfcb5db)
- **Extraction code:** `5EBx`
- **Evaluation input size:** `256 × 256`

The download is provided separately from this GitHub repository. Please verify the contents and permitted usage conditions before redistributing the dataset.

### 2. CattleSegment v7

- **Source:** [CattleSegment v7 on Roboflow Universe](https://universe.roboflow.com/cattledetector/cattlesegment-60nea/dataset/7)
- **Total:** 2,033 images
- **Official split:** 1,770 training / 165 validation / 98 test images
- **Original dataset task:** instance segmentation; the experimental masks must be prepared consistently for the binary cattle-body segmentation evaluation.

### 3. SideViewCows2026 — `snapshots` Subset

- **Source:** [SideViewCows2026 on Zenodo](https://doi.org/10.5281/zenodo.21605650)
- **Subset:** 607 images with corresponding masks
- **Experimental partition:** 429 training / 78 validation / 100 test images
- **Note:** Only the `snapshots` subset is used; these results do not represent evaluation on the entire SideViewCows2026 collection.

**Evaluation protocol:** The model is trained independently for each dataset and evaluated using the corresponding validation partition. The reported cross-source results are **not** zero-shot evaluations of a single Jinnan-trained checkpoint.

## Training Configuration

The main Jinnan dataset experiments use the following settings:

| Item | Configuration |
|:--|:--|
| Framework | PyTorch 2.1.1 |
| Python | 3.10.13 |
| CUDA / cuDNN | 11.8 / 8.7.0 |
| GPU | NVIDIA GeForce RTX 4090 D (24 GB) |
| Image size | 256 × 256 |
| Batch size | 4 |
| Maximum epochs | 200 |
| Loss function | Dice loss + Tversky loss |
| Checkpoint selection | Best validation Dice |


FLOPs are measured for a single forward pass at **256 × 256** resolution. Hardware throughput numbers, when provided, should be interpreted with their stated device and batch-size settings rather than as mobile-device inference performance.

## Code and Reproducibility

The public repository does not yet include a verified runnable training or inference entry point. To avoid providing misleading commands, setup and execution instructions will be documented after the matching source code, dependencies, data preparation, and checkpoint files are available.

For reproducible evaluation, the project will need to specify the dataset partitions, mask preprocessing, random seeds, best-checkpoint criterion, and metric implementations used in the experiments.

## Citation

If you use this work, please cite the paper after its bibliographic information becomes available:

> **DCENet: A dual-domain collaborative enhancement network for accurate and lightweight cattle segmentation in complex farm environments.** Manuscript information to be updated upon publication.

If you use the public datasets, please also cite their original dataset sources linked above.

## License and Data Use

A source-code license has **not yet been specified** in this repository. Do not assume that the project code or the collected dataset is distributed under an open-source license. Third-party datasets are subject to their respective license and attribution requirements.

## Contact

For questions or reproducibility-related requests, please open an [issue](https://github.com/LiuYiFei615/DCENet_cow/issues) in this repository.
