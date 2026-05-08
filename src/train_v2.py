"""
FILE: train_v2.py
DESCRIPTION: V2.1 Expert Training Pipeline - Specialized for Maritime Vessels.
             Optimized for the RTX 3060 (Laptop) using Single-Class OBB logic.
AUTHOR: Eduardo
DATE: January 2026
"""

from ultralytics import YOLO
from pathlib import Path
import os

# --- CONFIGURATION (Setup) ---
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODEL_VARIANT = "yolov8n-obb.pt"
DATA_CONFIG   = PROJECT_ROOT / "maritime_v2.yaml"
PROJECT_NAME  = str(PROJECT_ROOT / "runs/obb/AeroMarine-Surveillance-V2")
RUN_NAME      = "OBB-Vessel-Expert-V2-1"

def train_v2_expert():
    # 1. Initialize the base OBB model
    model = YOLO(MODEL_VARIANT)

    print(f"[START] Training Vessel Expert model...")
    print(f"[INFO] Targeting project: {PROJECT_NAME}/{RUN_NAME}")

    # 2. Fire the optimized engine
    model.train(
        # --- Data & Contract ---
        data=str(DATA_CONFIG),
        epochs=100,
        imgsz=640,
        project=PROJECT_NAME,
        name=RUN_NAME,

        # --- Hardware Optimization (RTX 3060 Laptop) ---
        device=0,           # GPU
        batch=-1,           # Auto-batch: Finds the limit of mine 6GB VRAM
        half=True,          # FP16 precision: 2x faster on Ampere architecture
        workers=8,          # Optimized for Arch Linux CPU threading
        cache=True,         # RAM caching for ultra-fast epoch cycles

        # --- Specialized Expert Tuning ---
        patience=25,        # Acceptable lower patience as single-class converges faster
        lr0=0.01,           # Starting learning rate
        cos_lr=True,        # Smooth decay for high-precision convergence
        weight_decay=0.0005,

        # --- Advanced Augmentations  ---
        #
        mosaic=1.0,         # Handles ships in crowded port environments
        mixup=0.10,         # Slightly reduced for single-class to avoid noise
        copy_paste=0.1,     # Helps the model see more ships per frame
        scale=0.5,          # Robustness for varying drone altitudes
        degrees=15.0,       # Rotation invariance for OBB

        # --- Output & Logging ---
        save=True,
        plots=True,
        overlap_mask=True,
        exist_ok=True       # Overwrites the folder if it exists
    )

if __name__ == "__main__":
    # Final Sanity Check: Path and Folder cleanup
    if not os.path.exists(DATA_CONFIG):
        print(f"[ERROR] {DATA_CONFIG} not found in root! Pipeline aborted.")
    else:
        # Check if the project folder exists and inform the user
        target_path = os.path.join(PROJECT_NAME, RUN_NAME)
        if os.path.exists(target_path):
            print(f"[WARN] Training folder '{target_path}' already exists.")
            print(f"[INFO] Files will be updated/replaced in that directory.")

        train_v2_expert()