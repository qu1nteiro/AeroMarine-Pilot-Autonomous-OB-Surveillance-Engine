AeroMarine-Pilot: Autonomous OBB Surveillance Engine
========================================================================

DESCRIPTION
-----------
The project, named "AeroMarine-Pilot", aims to be a high-performance 
computer vision engine specialized in the detection and orientation of maritime vessels. 
Built on the YOLOv8-OBB architecture, this project evolved from a satellite-based 
foundation (not uploaded to this repository) into a robust "Vessel Expert" model capable of processing 
multimodal aerial data (Drone & Satellite) with extreme precision.


KEY PERFORMANCE INDICATORS
----------------------------------------
Target Class: Vessel (Merged Ship & Boat)
Hardware: NVIDIA RTX 3060 (Laptop GPU)

- mAP50:           0.988 (Near-perfect detection)
- mAP50-95:        0.844 (High-fidelity orientation)
- Precision:       97.7% (Minimal false positives)
- Inference Speed: 5.5ms (~180 FPS)


CORE METHODOLOGY
----------------
1. Oriented Bounding Boxes (OBB):
   Captures exact heading and aspect ratio of vessels, minimizing 
   background noise (water/docks) compared to standard HBB.

2. Dataset Fusion:
   Trained on 12,000+ synchronized instances from:
   - HRSC2016 (Satellite imagery daset)
   - Zenodo Aerial (Tactical Drone footage dataset)

3. Refinement:
   The model was strategically optimized as a single-class "Vessel Expert" 
   to maximize neural capacity and convergence speed for maritime tasks.


PROJECT STRUCTURE
-----------------
aeromarine-pilot/
├── docs/               # Technical documentation and literature support
├── src/                # Core engine (Merger, Training, Showcase scripts)
├── maritime_v2.yaml    # Dataset configuration contract
├── .gitignore          # Exclusion rules (PyCharm/Large Data)
└── README.txt          # Project overview


USAGE
-----
1. Environment Setup:
   Ensure Ultralytics and PyTorch (CUDA supported) are installed.

2. Execution:
   To run the visual showcase on validation data:
   $ python src/showcase_vessel_expert.py


TECHNICAL REPORT
----------------
A detailed technical report is in preparation, covering coordinate 
normalization, hyperparameter tuning, and hardware optimization.


AUTHORS
-------
Project Autor: Eduardo
Status: V2 Expert Model (January 2026)
