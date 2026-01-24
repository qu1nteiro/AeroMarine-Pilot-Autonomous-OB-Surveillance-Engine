"""
FILE: showcase_vessel_expert.py
DESCRIPTION: Inference Pipeline for the Specialized Vessel Model.
             Generates high-resolution visual proof of model accuracy.
AUTHOR: Eduardo
DATE: January 2026
"""

import os
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION (Macros) ---
PROJECT_ROOT = Path("/home/eduardo/Documents/computacional/aia/aeromarine-pilot")
BEST_WEIGHTS = PROJECT_ROOT / "runs/obb/AeroMarine-Surveillance-V2/OBB-Vessel-Expert-V2-1/weights/best.pt"
VAL_IMAGES = PROJECT_ROOT / "data_v2/images/val"
OUTPUT_DIR = PROJECT_ROOT / "results/final_showcase"


def run_expert_showcase():
    print("[START] Generating Visual Showcase for Vessel Expert V2.1...")

    if not BEST_WEIGHTS.exists():
        print(f"[ERROR] Weights not found at: {BEST_WEIGHTS}")
        return

    # 1. Load the Expert Model
    model = YOLO(BEST_WEIGHTS)

    # 2. Select a subset of validation images (top 10 to avoid clutter)
    images = list(VAL_IMAGES.glob("*.jpg"))[:10]

    if not images:
        print(f"[ERROR] No validation images found in {VAL_IMAGES}")
        return

    # 3. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Run Inference
    for img_path in images:
        print(f"[LOG] Processing: {img_path.name}")

        # Inference
        results = model.predict(
            source=img_path,
            conf=0.4,  # High confidence for a clean showcase
            save=True,
            project=OUTPUT_DIR,
            name="detections",
            exist_ok=True,
            line_width=2,
            show_labels=True,
            show_conf=True
        )

    print(f"\n[SUCCESS] Showcase generated at: {OUTPUT_DIR}/detections")


if __name__ == "__main__":
    run_expert_showcase()