# Portrait Artwork Restoration (FaceArt Restoration) using Mask-Aware Transformer (MAT)

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D%201.7.1-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0%2B-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Research--Only-orange.svg)](LICENSE)

This repository contains the official implementation of the graduation thesis project: **Portrait Artwork Restoration (FaceArt Restoration) using Mask-Aware Transformer (MAT)**.

* **Author:** Ngô Thị Thương (Student ID: **ITCSIU21160**)
* **Advisor:** TS. Lê Thị Ngọc Hạnh
* **Affiliation:** School of Computer Science and Engineering, International University - VNU-HCM
* **Department:** Computer Science

---

## 📖 Project Overview

FaceArt Restoration focuses on restoring severely damaged, cracked, or missing areas in historical portrait paintings. Traditional image inpainting methods struggle with large-hole restorations in artistic styles. This project introduces a robust, multi-phase framework built on the state-of-the-art **Mask-Aware Transformer (MAT)** (CVPR 2022 Best Paper Finalist) to restore structural coherence and fine artistic textures to portrait artwork.

### Core Innovations & Architecture

The architecture optimizes the standard MAT model using a **Three-Phase Training & Architecture Framework**:

1. **Relative Position Bias & Additive Mask Bias:** Introduces relative spatial weights inside window-based attention layers to accommodate non-local textures, alongside mask-aware biases to guide attention away from invalid pixels.
2. **Deterministic Latent Gate:** Stabilizes the mixing of style features and latent representations to avoid mode collapse on art distributions.
3. **Multi-Scale Transformer Adapters:** Injectable adapter modules trained at $32 \times 32$ and $16 \times 16$ resolutions to preserve general MAT features while fitting the specific domain of portrait paintings.
4. **Structure Guidance & Adaptive Gate (Phase 3):** Embeds structural features of facial geometry (such as eyes, nose, mouth lines) into the decoding process, balanced by an adaptive gate to dynamically weigh structural consistency versus localized textures.

---

## 🖥️ Streamlit Interactive UI

The project features a premium Web UI powered by **Streamlit** for interactive, step-by-step restoration.

![Streamlit UI Architecture](figures/teasing.png)

### Key Features
* **Multi-Step Pipeline:** Progress seamlessly through `Input & Mask` $\to$ `Masked Input` $\to$ `Stage 1 (Coarse/Finetune)` $\to$ `Final Output (Refined/Adapter)`.
* **Interactive Canvas & Tools:** Freehand mask painting using `streamlit-drawable-canvas` along with direct geometric presets (`Cross`, `Rect`, `Scribble`).
* **Mask Nudge Adjustments:** Easily move, scale, and shift the mask alignment via direction buttons (`← Left`, `→ Right`, `↑ Up`, `↓ Down`) to center on facial damage.
* **Dual Backend Modes:** 
  * **Local:** Runs inference directly on the client machine (supports `cuda`, `cpu`, or `mps` for Apple Silicon).
  * **Remote:** Offloads computational workloads to a remote GPU server or Google Colab instance via a **FastAPI backend API** (`colab_inference_api.py`).
* **Visual Comparator & Lightbox:** Side-by-side comparative views of the Original, Stage 1, Stage 2 (Final), and original MAT Baseline outputs. Includes a zoom/pan Lightbox mode for inspecting brush strokes.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/thuongngo050902/THESIS.git
cd THESIS
```

### 2. Environment Setup (Recommended: Conda)
```bash
# Create environment with Python 3.8
conda create -n faceart_env python=3.8 -y
conda activate faceart_env
```
*(Alternatively, configure a standard Python venv)*

### 3. Install PyTorch
Select the command corresponding to your target hardware:

* **NVIDIA GPU (CUDA 11.8 - Recommended):**
  ```bash
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
  ```
* **Apple Silicon Macbook (MPS - M1/M2/M3):**
  ```bash
  pip install torch torchvision
  ```
* **CPU-Only:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> **Ninja Compiler:** `ninja` is required for building high-performance CUDA kernels during training.
> * Conda: `conda install -c conda-forge ninja -y`
> * Pip: `pip install ninja`
> If running on CPU-only or macOS (MPS), if compilation fails, the code defaults automatically to CPU fallbacks.

---

## 🚀 How to Run the Application

### 1. Launching the Web UI
Run the following command to start the Streamlit application:
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```
Open [http://127.0.0.1:8501](http://127.0.0.1:8501) on your web browser.

### 2. Running the Remote API Backend
To run the server backend on a remote machine equipped with GPU acceleration (or Colab):
```bash
uvicorn colab_inference_api:app --host 0.0.0.0 --port 8000
```
On the Streamlit UI, open **Advanced Controls** in the sidebar, switch `Backend Mode` to `remote`, and paste the host address (e.g. `http://<your-server-ip>:8000` or the generated ngrok URL).

### 3. Command-Line Interface (CLI) Batch Inference
To run inference directly through the command-line on a directory of images:
```bash
python generate_image.py \
    --network checkpoints/network-snapshot-000072.pkl \
    --dpath test_sets/CelebA-HQ/images \
    --mpath test_sets/CelebA-HQ/masks \
    --outdir samples_output
```
*Note: Input images must have dimensions that are multiples of 512. Standard sizes like 512x512 are fully supported.*

---

## 🏋️ Model Training Workflow

Model training follows a strict **3-Phase training process** to ensure fine details and global structures are learned in stages.

### 1. Dataset Preparation
Pack images into an uncompressed ZIP format to accelerate GPU I/O operations:
```bash
python dataset_tool.py --source /path/to/raw/images --dest datasets/faceart_train.zip --width 512 --height 512
```

### 2. Training Stages

#### ⚡ Phase 1: Base Finetuning
Finetunes the base MAT network on the custom portrait artwork dataset using relative biases and deterministic gate configurations.
```bash
python train.py \
    --outdir=runs/faceart_phase1_relbias_gate \
    --data=datasets/faceart_train.zip \
    --data_val=datasets/faceart_val.zip \
    --cfg=places512 \
    --resume=checkpoints/Places_512_FullData.pkl \
    --epochs=40 \
    --batch=4 \
    --lr=5e-5 \
    --enable-rel-pos-bias=True \
    --enable-mask-bias=True \
    --enable-deterministic-latent-gate=True
```

#### ⚡ Phase 2: Transformer Adapter Training
Freezes core network parameters and trains lightweight adapter layers at $32 \times 32$ and $16 \times 16$ scales.
```bash
python train.py \
    --outdir=runs/faceart_phase2_tran_adapter \
    --data=datasets/faceart_train.zip \
    --data_val=datasets/faceart_val.zip \
    --cfg=places512 \
    --resume=runs/faceart_phase1_relbias_gate/00000-places512/network-snapshot-000040.pkl \
    --epochs=30 \
    --batch=4 \
    --lr=2.5e-5 \
    --enable-rel-pos-bias=True \
    --enable-mask-bias=True \
    --enable-deterministic-latent-gate=True \
    --enable-tran-adapter-32=True \
    --enable-tran-adapter-16=True
```

#### ⚡ Phase 3: Structure Guidance & Adaptive Gate Training
Enables geometry-guided structures and the adaptive gate module to compile the final outputs.
```bash
python train.py \
    --outdir=runs/faceart_phase3_adaptive_structure_guidance \
    --data=datasets/faceart_train.zip \
    --data_val=datasets/faceart_val.zip \
    --cfg=places512 \
    --resume=runs/faceart_phase2_tran_adapter/00000-places512/network-snapshot-000030.pkl \
    --epochs=10 \
    --batch=4 \
    --lr=5e-5 \
    --enable-rel-pos-bias=True \
    --enable-mask-bias=True \
    --enable-deterministic-latent-gate=True \
    --enable-tran-adapter-32=True \
    --enable-tran-adapter-16=True \
    --enable-structure-guidance=True \
    --enable-structure-fuse-16=True \
    --enable-structure-fuse-stage2=True \
    --enable-adaptive-structure-gate=True
```

*Note: Automation training scripts for tmux and background training are provided in [collab/](file:///d:/2025-2026/Thesis/Clone/MAT/collab/).*

---

## 📊 Quantitative Metrics Evaluation

To calculate standard image quality evaluation metrics, use the scripts provided in the `evaluation/` directory:

1. **PSNR, SSIM, and L1 Loss:**
   ```bash
   python evaluation/eval_psnr_ssim.py --rec_path /path/to/reconstructed/images --gt_path /path/to/ground_truth/images
   ```
2. **LPIPS (Learned Perceptual Image Patch Similarity):**
   ```bash
   python evaluation/eval_lpips.py --rec_path /path/to/reconstructed/images --gt_path /path/to/ground_truth/images
   ```
3. **FID (Fréchet Inception Distance):**
   ```bash
   python evaluation/eval_fid.py --rec_path /path/to/reconstructed/images --gt_path /path/to/ground_truth/images
   ```

---

## 📜 Citation & Acknowledgements

This thesis project builds upon the foundations of the following research papers:

```bibtex
@inproceedings{li2022mat,
    title={MAT: Mask-Aware Transformer for Large Hole Image Inpainting},
    author={Li, Wenbo and Lin, Zhe and Zhou, Kun and Qi, Lu and Wang, Yi and Jia, Jiaya},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2022}
}

@inproceedings{karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila and Samuli Laine},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```

---

## 📄 Vietnamese Documentation

For a detailed step-by-step installation, Google Colab/FastAPI server connection, Streamlit Web UI controls, and model retraining instructions written in Vietnamese, please refer to:  
👉 **[Installation & Operation Guide (Vietnamese)](file:///d:/2025-2026/Thesis/Clone/MAT/ITCSIU21160_NgoThiThuong.md)**
