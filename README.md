# PFTNet: Progressive Focused Transformer Network for UAV Image Dehazing

**By** **Jiangong Xu**, **Weibao Xue**, **Jun Pan**, **and Mi Wang**

This repository provides the **official PyTorch implementation** of **PFTNet (Progressive Focused Transformer Network)**, a Transformer-based image restoration framework designed for **UAV aerial image dehazing under spatially non-uniform atmospheric degradation**.

Unlike conventional CNN- or Transformer-based dehazing methods that rely on static attention patterns or implicit degradation modeling, PFTNet introduces a **progressive attention focusing paradigm** and explicitly integrates **degradation-aware modeling** and **deformable sparse attention**, achieving improved restoration fidelity and geometric consistency in challenging UAV scenarios.

---

<p align="center">
  <img src="images/Fig_1.png" width="90%">
</p>

<p align="center">
  <em>Figure 1. Overall architecture of the proposed PFTNet, illustrating the progressive focused attention mechanism and the integration of degradation-aware and deformable sparse attention modules.</em>
</p>

---


## 🔍 Overview

UAV aerial images are often affected by **spatially varying haze**, where degradation strength changes significantly with scene depth and viewing geometry. Existing methods typically suffer from:

- Insufficient global-to-local modeling capability
- Mismatch between attention patterns and scene geometry
- Blurred distant regions and structural distortions

PFTNet addresses these issues by progressively narrowing the attention scope from global context perception to local structure refinement, while explicitly guiding attention using degradation cues and deformable geometric alignment.

---

## ✨ Key Contributions

- **Progressive Focused Attention**
  - Attention range gradually contracts across network depth, enabling a smooth transition from global degradation modeling to local detail refinement.

- **Degradation-Aware Attention Module (DAAM)**
  - Explicitly models spatially non-uniform degradation using frequency-domain difference cues, separating degradation variations from content changes.

- **Deformable Sparse-Aware Attention Module (DSAAM)**
  - Adopts deformable attention sampling to align attention with elongated and irregular structures (e.g., roads, building edges), reducing geometric distortion.

- **Efficient Multi-Scale Transformer Backbone**
  - U-Net–style encoder–decoder architecture with flexible model scaling (PFTNet-S / PFTNet-L).

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
```

## 🛠 Requirements

· Python ≥ 3.8  
· PyTorch ≥ 1.12  
· torchvision  
· numpy  
· Pillow  

Install dependencies:

```bash
pip install torch torchvision numpy pillow
```

## 🚀 Training
**Dataset Preparation**

Prepare paired hazy and clear images with identical filenames:

```text
train_hazy/
├── 0001.png
├── 0002.png
└── ...

train_clear/
├── 0001.png
├── 0002.png
└── ...
```
**Train PFTNet-L**
```text
python train.py \
  --model pftnet-l \
  --train_hazy path/to/train_hazy \
  --train_clear path/to/train_clear \
  --batch_size 12 \
  --epochs 150 \
  --lr 2e-4
```
**Train PFTNet-S (Lightweight)**
```text
python train.py \
  --model pftnet-s \
  --train_hazy path/to/train_hazy \
  --train_clear path/to/train_clear
```
## 📐 Model Variants

| Model    | Embed Dim | Encoder Depths | Decoder Depths |
|----------|-----------|----------------|----------------|
| PFTNet-S | 48        | [1, 2, 2, 2]   | [1, 1, 1]      |
| PFTNet-L | 64        | [2, 2, 4, 4]   | [2, 2, 2]      |

## 🔎 Inference
```text
python infer.py \
  --model pftnet-l \
  --ckpt checkpoints/pftnet-l_epoch_150.pth \
  --input_dir path/to/hazy_images \
  --output_dir path/to/results
```
The restored images will be saved to **output_dir**.

## 🧪 Experimental Datasets

All experiments in this work are conducted on publicly available UAV and aerial
image datasets:

- **DroneVehicle**  
  https://github.com/VisDrone/DroneVehicle

- **RTDOD (Real-Time Drone Object Detection Dataset)**  
  https://github.com/fenght96/RTDOD

- **HazyDet**  
  https://github.com/GrokCV/HazyDet

These datasets provide diverse UAV scenes with varying atmospheric conditions, viewing geometries, and object distributions, making them suitable for evaluating dehazing robustness and generalization.

## 📎 Citation

If you find this work useful, please consider citing:

```bibtex
@article{PFTNet2026,
  title   = {PFTNet: Progressive Focused Transformer Network for UAV Image Dehazing},
  author  = {Xu, Jiangong and Xue, Weibao and Pan, Jun and Wang, Mi},
  journal = {Journal Name},
  year    = {2026},
  note    = {Under review}
}
```

---


## 📄 License

This project is released under the **MIT License**.  
See `LICENSE` for more details.

