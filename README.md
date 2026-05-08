# AeroMarine-Pilot

Vessel detection in satellite and drone imagery using YOLOv8n-OBB (Oriented Bounding Boxes).  
Single-class model trained to locate ships and boats at arbitrary orientations across heterogeneous aerial datasets.

<!-- Replace with a real detection sample: ![Detection sample](assets/sample_detection.jpg) -->

---

## Results

Evaluated on 2,332 validation images. Full log: [`results_eval.txt`](results_eval.txt).

| Metric | Value |
|---|---|
| Precision | 0.977 |
| Recall | 0.969 |
| mAP@0.50 | 0.988 |
| mAP@0.50:0.95 | 0.844 |
| Inference (batched, GPU) | 1.9 ms/img |
| Latency (single image) | 12.4 ms |
| Throughput (single image) | ~81 FPS |

Training curves and confusion matrix: [`runs/obb/AeroMarine-Surveillance-V2/OBB-Vessel-Expert-V2-1/`](runs/obb/AeroMarine-Surveillance-V2/OBB-Vessel-Expert-V2-1/)

---

## Overview

Maritime vessel detection from above is a non-trivial detection problem: ships appear at arbitrary rotations, vary enormously in scale (from fishing boats to container ships), and are often densely packed in ports. Axis-aligned bounding boxes lose most of the geometric signal; OBB regression preserves it.

This project:
1. Merges two publicly available aerial datasets — HRSC2016 (satellite, high-resolution) and a Zenodo aerial drone dataset — into a single unified split.
2. Collapses the multi-class labels (ship, boat) into a single **Vessel** class, removing person detections that are irrelevant for maritime surveillance.
3. Converts all horizontal bounding boxes to 8-point OBB format for consistent label representation.
4. Fine-tunes YOLOv8n-OBB for 100 epochs with FP16 precision on a consumer GPU.

The nano variant was chosen deliberately: it runs in real-time on embedded hardware while still achieving >98% mAP@0.50.

---

## Installation

**Requirements:** Python 3.12, CUDA 12.8 (adjust the PyTorch index URL for other CUDA versions).

```bash
git clone https://github.com/qu1nteiro/AeroMarine-Pilot-Autonomous-OBB-Surveillance-Engine.git
cd AeroMarine-Pilot-Autonomous-OBB-Surveillance-Engine

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# PyTorch with CUDA 12.8 — change the URL for other CUDA versions or CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Usage

### 1 — Prepare the dataset

Download the two source datasets and place them as follows:

```
archive/v1/data/processed/   ← HRSC2016 (pre-converted to YOLO OBB format)
data_raw/zenodo_aerial/      ← Zenodo aerial dataset
```

Then merge them into the unified V2 split:

```bash
python src/v2_dataset_merger.py
# Outputs to data_v2/  (9,333 train / 2,332 val images)
```

### 2 — Train

```bash
python src/train_v2.py
# Weights saved to runs/obb/AeroMarine-Surveillance-V2/OBB-Vessel-Expert-V2-1/weights/
```

Training takes approximately 4 hours on an RTX 3060 Laptop (100 epochs, FP16, auto-batch).

### 3 — Evaluate

```bash
python src/evaluate.py
# Prints metrics and saves results_eval.txt
```

### 4 — Inference (visual showcase)

```bash
python src/showcase_vessel_expert.py
# Saves annotated images to results/final_showcase/detections/
```

---

## Datasets

| Dataset | Source | Images used | Classes |
|---|---|---|---|
| HRSC2016 | [Liu et al., 2017](https://doi.org/10.1117/12.2266540) — satellite imagery | ~7,500 | Ship |
| Zenodo Aerial | Drone aerial imagery | ~4,100 | Boat, Ship (Person excluded) |

Both datasets are used for research purposes only. Refer to the original publications and dataset pages for licensing terms.

**Label unification:** all vessel instances (ship, boat) are mapped to a single `Vessel` class (id 0). The Zenodo test split is folded into the validation set; no held-out test set is maintained.

---

## Repository structure

```
aeromarine-pilot/
├── src/
│   ├── train_v2.py              # Training pipeline
│   ├── evaluate.py              # Validation metrics + speed benchmark
│   ├── showcase_vessel_expert.py# Visual inference on validation images
│   └── v2_dataset_merger.py     # HRSC2016 + Zenodo → unified V2 dataset
├── runs/
│   └── obb/AeroMarine-Surveillance-V2/OBB-Vessel-Expert-V2-1/
│       ├── weights/best.pt      # Best checkpoint (not versioned — train to generate)
│       ├── results.csv          # Per-epoch training metrics
│       └── results.png          # Training curves
├── maritime_v2.yaml             # Dataset config for Ultralytics
├── requirements.txt
└── results_eval.txt             # Reproducible evaluation output
```

Model weights (`*.pt`) are excluded from version control. Run `train_v2.py` to generate them, or contact the author for a pre-trained checkpoint.

---

## Hardware

Trained and benchmarked on:

- **GPU:** NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
- **Precision:** FP16 (Ampere tensor cores)
- **Framework:** Ultralytics 8.4.7, PyTorch 2.9.1+cu128

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv8n-OBB |
| Parameters | 3.08 M |
| GFLOPs | 8.3 |
| Input size | 640 × 640 |
| Output | 8-point OBB (x1 y1 … x4 y4) + confidence |
| Classes | 1 (Vessel) |
