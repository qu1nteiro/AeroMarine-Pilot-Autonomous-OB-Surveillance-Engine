"""
FILE: evaluate.py
DESCRIPTION: Standalone evaluation and benchmark script for AeroMarine-Pilot.
             Runs model.val() on the validation set and measures inference speed.
             Saves a results_eval.txt to the project root for reproducibility.
USAGE: python src/evaluate.py
"""

import time
from pathlib import Path

from ultralytics import YOLO

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS      = PROJECT_ROOT / "runs/obb/AeroMarine-Surveillance-V2/OBB-Vessel-Expert-V2-1/weights/best.pt"
DATA_CONFIG  = PROJECT_ROOT / "maritime_v2.yaml"
OUTPUT_FILE  = PROJECT_ROOT / "results_eval.txt"

BENCHMARK_RUNS = 200  # number of predict() calls for speed measurement


def validate(model):
    """Run model.val() and return the metrics object."""
    print("[INFO] Running validation on the full validation set...")
    metrics = model.val(data=str(DATA_CONFIG), verbose=False)
    return metrics


def benchmark_speed(model, val_dir):
    """Warm-up + timed inference on a single image. Returns mean ms and FPS."""
    images = sorted(val_dir.glob("*.jpg"))
    if not images:
        images = sorted(val_dir.glob("*.JPG"))
    if not images:
        return None, None

    target = str(images[0])

    # Warm-up: 10 runs (fills CUDA cache, JIT compilation)
    for _ in range(10):
        model.predict(source=target, verbose=False, save=False)

    # Timed runs
    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        model.predict(source=target, verbose=False, save=False)
    elapsed = time.perf_counter() - start

    mean_ms  = (elapsed / BENCHMARK_RUNS) * 1000
    fps      = 1000 / mean_ms
    return mean_ms, fps


def format_report(metrics, mean_ms, fps):
    rd = metrics.results_dict
    sp = metrics.speed  # {'preprocess': x, 'inference': y, 'postprocess': z} in ms/image

    lines = [
        "=" * 50,
        "  AeroMarine-Pilot — Evaluation Results",
        "=" * 50,
        "",
        "[ Validation Metrics ]",
        f"  Precision      : {rd['metrics/precision(B)']:.4f}",
        f"  Recall         : {rd['metrics/recall(B)']:.4f}",
        f"  mAP@0.50       : {rd['metrics/mAP50(B)']:.4f}",
        f"  mAP@0.50-0.95  : {rd['metrics/mAP50-95(B)']:.4f}",
        "",
        "[ Inference Speed (from model.val) ]",
        f"  Preprocess     : {sp['preprocess']:.2f} ms/img",
        f"  Inference      : {sp['inference']:.2f} ms/img",
        f"  Postprocess    : {sp['postprocess']:.2f} ms/img",
    ]

    if mean_ms is not None:
        lines += [
            "",
            f"[ Standalone Benchmark ({BENCHMARK_RUNS} runs, single image) ]",
            f"  Mean latency   : {mean_ms:.2f} ms",
            f"  Throughput     : {fps:.1f} FPS",
        ]

    lines += [
        "",
        "[ Configuration ]",
        f"  Weights        : {WEIGHTS.relative_to(PROJECT_ROOT)}",
        f"  Dataset config : {DATA_CONFIG.name}",
        "=" * 50,
    ]
    return "\n".join(lines)


def main():
    if not WEIGHTS.exists():
        print(f"[ERROR] Weights not found: {WEIGHTS}")
        return
    if not DATA_CONFIG.exists():
        print(f"[ERROR] Dataset config not found: {DATA_CONFIG}")
        return

    model = YOLO(str(WEIGHTS))

    metrics  = validate(model)
    val_dir  = PROJECT_ROOT / "data_v2/images/val"
    mean_ms, fps = benchmark_speed(model, val_dir)

    report = format_report(metrics, mean_ms, fps)

    print("\n" + report)

    with open(OUTPUT_FILE, "w") as f:
        f.write(report + "\n")

    print(f"\n[INFO] Report saved to: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
