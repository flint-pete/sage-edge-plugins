#!/usr/bin/env python3
"""
YOLO Integration Test — Real Model Inference on DGX Spark
Downloads yolo11x.pt, runs real inference on test images,
verifies detections and pywaggle publish/upload.

Requires: GPU with CUDA, ultralytics, pywaggle, opencv, torch
"""
import json
import os
import sys
import time
import tempfile
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── pywaggle local output setup ─────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "yolo-integration")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["PYWAGGLE_LOG_DIR"] = OUTPUT_DIR

from waggle.plugin import Plugin

# ── paths ────────────────────────────────────────────────────────────
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample-images")
TEST_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "test-images", "yolo")

# Use real test images if available, fall back to synthetic sample-images
if os.path.isdir(TEST_IMAGE_DIR):
    _test_imgs = sorted(
        f for f in os.listdir(TEST_IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
else:
    _test_imgs = []

if _test_imgs:
    IMAGE_DIR = TEST_IMAGE_DIR
    IMAGES = _test_imgs
    print(f"Using {len(IMAGES)} real test image(s) from {TEST_IMAGE_DIR}")
else:
    IMAGE_DIR = SAMPLE_DIR
    IMAGES = ["urban_street.jpg", "wildlife.jpg", "sky_clouds.jpg"]
    print(f"No real test images in {TEST_IMAGE_DIR}, using synthetic samples")

MODEL_NAME = "yolo11x.pt"


def draw_boxes(frame, detections):
    """Draw bounding boxes on frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return annotated


def main():
    print("=" * 70)
    print("YOLO INTEGRATION TEST — Real Model Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory
        if mem_total > 0:
            print(f"GPU Memory: {mem_total / 1e9:.1f} GB")
        else:
            print("GPU Memory: unified memory (reported as 0)")
    print("=" * 70)

    # ── 1. Load model ────────────────────────────────────────────────
    print(f"\n[1/4] Loading {MODEL_NAME}...")
    t0 = time.time()
    model = YOLO(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.2f}s on {device}")
    print(f"  Classes: {len(model.names)}")
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

    # ── 2. Warmup ────────────────────────────────────────────────────
    print("\n[2/4] Warmup inference...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    t0 = time.time()
    model(dummy, verbose=False)
    warmup_time = time.time() - t0
    print(f"  Warmup done in {warmup_time:.2f}s")

    # ── 3. Run inference on test images ──────────────────────────────
    print(f"\n[3/4] Running inference on {len(IMAGES)} test images from {IMAGE_DIR}...")
    all_results = []

    for img_name in IMAGES:
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  SKIP {img_name} — not found")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  SKIP {img_name} — failed to read")
            continue

        h, w = frame.shape[:2]

        t0 = time.time()
        results = model(frame, conf=0.25, iou=0.45, verbose=False)
        inf_time = time.time() - t0

        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "class": cls_name,
                    "confidence": float(box.conf[0]),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        # Count per class
        counts = {}
        for det in detections:
            counts[det["class"]] = counts.get(det["class"], 0) + 1

        result = {
            "image": img_name,
            "resolution": f"{w}x{h}",
            "inference_ms": round(inf_time * 1000, 1),
            "total_detections": len(detections),
            "class_counts": counts,
            "detections": detections,
        }
        all_results.append(result)

        print(f"\n  {img_name} ({w}x{h})")
        print(f"    Inference: {inf_time*1000:.1f}ms")
        print(f"    Detections: {len(detections)}")
        for cls, cnt in sorted(counts.items()):
            print(f"      {cls}: {cnt}")

    # ── 4. Publish via pywaggle ──────────────────────────────────────
    print(f"\n[4/4] Publishing results via pywaggle...")
    published = 0
    uploaded = 0

    with Plugin() as plugin:
        for result in all_results:
            img_name = result["image"]
            img_path = os.path.join(IMAGE_DIR, img_name)
            frame = cv2.imread(img_path)
            ts = time.time_ns()

            # Publish per-class counts
            for cls_name, count in result["class_counts"].items():
                topic = f"env.count.{cls_name}"
                plugin.publish(
                    topic, count,
                    timestamp=ts,
                    meta={"camera": "test", "model": MODEL_NAME},
                )
                published += 1

            # Publish total
            plugin.publish(
                "env.count.total",
                result["total_detections"],
                timestamp=ts,
                meta={"camera": "test", "model": MODEL_NAME},
            )
            published += 1

            # Upload annotated image
            if result["detections"]:
                annotated = draw_boxes(frame, result["detections"])
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=OUTPUT_DIR)
                cv2.imwrite(tmp.name, annotated)
                plugin.upload_file(tmp.name, timestamp=ts,
                                   meta={"camera": "test",
                                         "detections": str(result["total_detections"])})
                uploaded += 1

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Model load time: {load_time:.2f}s")
    print(f"Warmup time: {warmup_time:.2f}s")
    print(f"Images processed: {len(all_results)}")

    total_dets = sum(r["total_detections"] for r in all_results)
    avg_ms = sum(r["inference_ms"] for r in all_results) / max(len(all_results), 1)
    print(f"Total detections: {total_dets}")
    print(f"Avg inference time: {avg_ms:.1f}ms")
    print(f"Published measurements: {published}")
    print(f"Uploaded images: {uploaded}")

    # Check pywaggle output
    ndjson_path = os.path.join(OUTPUT_DIR, "data.ndjson")
    if os.path.exists(ndjson_path):
        with open(ndjson_path) as f:
            lines = f.readlines()
        print(f"NDJSON records: {len(lines)}")
    else:
        print("WARNING: No data.ndjson found!")

    uploads_dir = os.path.join(OUTPUT_DIR, "uploads")
    if os.path.exists(uploads_dir):
        upload_files = os.listdir(uploads_dir)
        print(f"Upload files: {len(upload_files)}")
    else:
        print("WARNING: No uploads/ directory found!")

    # ── Pass/Fail ────────────────────────────────────────────────────
    passed = True
    if total_dets == 0:
        print("\nFAIL: No detections at all!")
        passed = False
    if published == 0:
        print("\nFAIL: No measurements published!")
        passed = False
    if uploaded == 0:
        print("\nFAIL: No images uploaded!")
        passed = False

    if passed:
        print("\n✓ YOLO INTEGRATION TEST PASSED")
    else:
        print("\n✗ YOLO INTEGRATION TEST FAILED")
        sys.exit(1)

    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "device": device,
            "load_time_s": round(load_time, 2),
            "warmup_time_s": round(warmup_time, 2),
            "images": all_results,
            "total_detections": total_dets,
            "avg_inference_ms": round(avg_ms, 1),
            "published": published,
            "uploaded": uploaded,
        }, f, indent=2)
    print(f"\nDetailed results: {results_path}")


if __name__ == "__main__":
    main()
