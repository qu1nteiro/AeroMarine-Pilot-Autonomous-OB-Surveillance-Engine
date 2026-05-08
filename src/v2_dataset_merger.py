"""
FILE: v2_dataset_merger.py
DESCRIPTION: Specialized Maritime Merger for HRSC2016 (V1) and Zenodo Aerial.
             Single-class focus (Vessel)
AUTHOR: Eduardo
DATE: January 2026
"""

import os
import shutil
from pathlib import Path

# --- CONFIGURATION (Macros) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_BASE      = PROJECT_ROOT / "archive/v1/data/processed"
ZENODO_BASE  = PROJECT_ROOT / "data_raw/zenodo_aerial"
OUTPUT_V2    = PROJECT_ROOT / "data_v2"

# Extension Support
IMG_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png", ".JPG", ".JPEG", ".BMP"}

# --- SINGLE CLASS MAPPING ---
# Removed class "0" (Person).
# If a class ID is not in this map, hbb_to_obb_8point returns None and skips it.
CLASS_MAP = {
    "1": "0",  # Zenodo Ship -> Vessel (0)
    "2": "0"   # Zenodo Boat -> Vessel (0)
}

# Split Mapping
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "val"
}


def hbb_to_obb_8point(line):
    """ Converts HBB (cx, cy, w, h) to OBB 8-points (x1 y1 ... x4 y4). """
    parts = line.strip().split()
    if len(parts) < 5: return None

    # This .get() returns None for the "0" (Person) class
    target_cls = CLASS_MAP.get(parts[0])

    # If it's a person (not in map), return None and ignore the label
    if target_cls is None:
        return None

    cx, cy, w, h = map(float, parts[1:])

    # 8-point calculation (Non-rotated rectangle)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2

    return f"{target_cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def migrate_v1():
    """ Migrates V1 OBB dataset. """
    print(f"[LOG] Migrating V1 from: {V1_BASE}")
    count = 0
    for split in ["train", "val"]:
        img_src = V1_BASE / "images" / split
        lbl_src = V1_BASE / "labels" / split

        if not img_src.exists(): continue

        for img_file in img_src.iterdir():
            if img_file.suffix in IMG_EXTENSIONS:
                shutil.copy(img_file, OUTPUT_V2 / "images" / split / img_file.name)
                lbl_file = lbl_src / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    shutil.copy(lbl_file, OUTPUT_V2 / "labels" / split / lbl_file.name)
                count += 1
    print(f"[SUCCESS] V1 Migration: {count} images.")


def process_zenodo():
    """ Processes Zenodo, filtering for Vessels only. """
    print(f"[LOG] Processing Zenodo from: {ZENODO_BASE}")
    count = 0
    for src_split, target_split in SPLIT_MAP.items():
        img_dir = ZENODO_BASE / src_split
        lbl_dir = ZENODO_BASE / "Annotations/Yolo" / src_split

        if not img_dir.exists():
            print(f"[WARN] Folder not found: {img_dir}")
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix in IMG_EXTENSIONS:
                # 1. Copy Image
                shutil.copy(img_file, OUTPUT_V2 / "images" / target_split / img_file.name)

                # 2. Convert HBB to OBB (Filtering active here)
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                obb_lines = []
                if lbl_file.exists():
                    with open(lbl_file, 'r') as f:
                        for line in f:
                            converted = hbb_to_obb_8point(line)
                            if converted:
                                obb_lines.append(converted)

                # 3. Save (If obb_lines is empty, it's a valid background image)
                with open(OUTPUT_V2 / "labels" / target_split / f"{img_file.stem}.txt", 'w') as f:
                    f.write("\n".join(obb_lines))
                count += 1
    print(f"[SUCCESS] Zenodo Processing: {count} images.")


def main():
    # Setup Workspace
    if OUTPUT_V2.exists():
        answer = input(f"[WARN] '{OUTPUT_V2}' already exists. Delete and rebuild? [y/N] ").strip().lower()
        if answer != "y":
            print("[ABORT] Operation cancelled.")
            return
        print(f"[LOG] Cleaning up old directory: {OUTPUT_V2}")
        shutil.rmtree(OUTPUT_V2)

    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        (OUTPUT_V2 / folder).mkdir(parents=True, exist_ok=True)

    print("--- STARTING V2 VESSEL EXPERT PIPELINE ---")
    migrate_v1()
    process_zenodo()
    print(f"--- COMPLETE: Dataset V2 (Vessels Only) ready at {OUTPUT_V2} ---")


if __name__ == "__main__":
    main()