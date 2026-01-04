# PFTNet: Progressive Focused Transformer Network for UAV Image Dehazing

This repository provides the **official PyTorch implementation** of  
**PFTNet (Progressive Focused Transformer Network)**, a Transformer-based
image restoration framework designed for **UAV aerial image dehazing under
spatially non-uniform atmospheric degradation**.

Unlike conventional CNN- or Transformer-based dehazing methods that rely on
static attention patterns or implicit degradation modeling, PFTNet introduces
a **progressive attention focusing paradigm** and explicitly integrates
**degradation-aware modeling** and **deformable sparse attention**, achieving
improved restoration fidelity and geometric consistency in challenging UAV
scenarios.

---

## 🔍 Overview

UAV aerial images are often affected by **spatially varying haze**, where
degradation strength changes significantly with scene depth and viewing
geometry. Existing methods typically suffer from:

- Insufficient global-to-local modeling capability
- Mismatch between attention patterns and scene geometry
- Blurred distant regions and structural distortions

PFTNet addresses these issues by progressively narrowing the attention scope
from global context perception to local structure refinement, while explicitly
guiding attention using degradation cues and deformable geometric alignment.

---

## ✨ Key Contributions

- **Progressive Focused Attention**
  - Attention range gradually contracts across network depth, enabling a smooth
    transition from global degradation modeling to local detail refinement.

- **Degradation-Aware Attention Module (DAAM)**
  - Explicitly models spatially non-uniform degradation using frequency-domain
    difference cues, separating degradation variations from content changes.

- **Deformable Sparse-Aware Attention Module (DSAAM)**
  - Adopts deformable attention sampling to align attention with elongated and
    irregular structures (e.g., roads, building edges), reducing geometric
    distortion.

- **Efficient Multi-Scale Transformer Backbone**
  - U-Net–style encoder–decoder architecture with flexible model scaling
    (PFTNet-S / PFTNet-L).

---

## 📂 Repository Structure

```text
PFTNet/
├── models/
│   ├── pftnet.py        # Main network definition
│   ├── pfam.py          # Progressive Focused Attention Module
│   ├── daam.py          # Degradation-Aware Attention Module
│   ├── dsaam.py         # Deformable Sparse-Aware Attention Module
│   ├── layers.py        # RSLN, MLP, and utility layers
│
├── ops/
│   └── deform_attn.py   # Core deformable attention sampling
│
├── train.py             # Training script
├── infer.py             # Inference script
└── README.md
