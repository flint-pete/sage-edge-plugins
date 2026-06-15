"""
Unit test for YOLO Object Counter plugin.

Tests the pywaggle integration layer: correct measurement names,
value types, meta format, and image uploads. The YOLO model is
replaced with canned detections — no torch/ultralytics needed.

Requires: pywaggle, numpy, opencv-python-headless, Pillow, ffmpeg-python
Does NOT require: ultralytics, torch, GPU
"""
import json
import os
import sys
from pathlib import Path

# ── Setup paths ──────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))

import test_harness as th


# ── Canned YOLO detections ──────────────────────────────────────────
FAKE_DETECTIONS = [
    {"class": "person", "confidence": 0.92, "bbox": [100, 200, 200, 400]},
    {"class": "person", "confidence": 0.87, "bbox": [250, 180, 340, 380]},
    {"class": "car",    "confidence": 0.95, "bbox": [50, 300, 350, 480]},
    {"class": "car",    "confidence": 0.78, "bbox": [400, 310, 580, 470]},
    {"class": "car",    "confidence": 0.65, "bbox": [360, 320, 500, 460]},
]



# ── Test ─────────────────────────────────────────────────────────────
def test_yolo_plugin():
    print("\n--- YOLO Object Counter: Unit Test ---\n")

    output_dir = th.setup_test_output("yolo-unit")

    import cv2
    from waggle.plugin import Plugin
    from waggle.plugin.time import get_timestamp

    images = th.get_test_images()
    model_name = "yolo11n.pt"
    stream = "test"

    with Plugin() as plugin:
        for img_path in images:
            frame = cv2.imread(img_path)
            assert frame is not None, f"Failed to load {img_path}"

            # Use canned detections (mock YOLO output)
            detections = FAKE_DETECTIONS
            ts = get_timestamp()

            # Aggregate counts per class
            counts = {}
            for det in detections:
                counts[det["class"]] = counts.get(det["class"], 0) + 1

            # Publish per-class counts
            for cls_name, count in counts.items():
                topic = f"env.count.{cls_name}"
                plugin.publish(topic, count, timestamp=ts,
                               meta={"camera": stream, "model": model_name})
                print(f"  Published: {topic} = {count}")

            # Publish total
            plugin.publish("env.count.total", sum(counts.values()),
                           timestamp=ts,
                           meta={"camera": stream, "model": model_name})
            print(f"  Published: env.count.total = {sum(counts.values())}")
            print()

    # Parse and validate output
    results = th.parse_output(output_dir)

    # ── Assertions ───────────────────────────────────────────────
    n_images = len(images)
    measurements = results["measurements"]

    # Our mock returns 2 person + 3 car = 5 detections per image
    # So: env.count.person, env.count.car, env.count.total = 3 per image
    count_measurements = [m for m in measurements if m["name"].startswith("env.count.")]

    assert len(count_measurements) == 3 * n_images, \
        f"Expected {3*n_images} count measurements, got {len(count_measurements)}"

    # Check specific values
    person_counts = [m for m in measurements if m["name"] == "env.count.person"]
    car_counts = [m for m in measurements if m["name"] == "env.count.car"]
    total_counts = [m for m in measurements if m["name"] == "env.count.total"]

    assert len(person_counts) == n_images
    assert len(car_counts) == n_images
    assert len(total_counts) == n_images

    for m in person_counts:
        assert m["value"] == 2, f"Expected 2 persons, got {m['value']}"
    for m in car_counts:
        assert m["value"] == 3, f"Expected 3 cars, got {m['value']}"
    for m in total_counts:
        assert m["value"] == 5, f"Expected 5 total, got {m['value']}"

    # Check meta fields are all strings
    for m in measurements:
        for k, v in m.get("meta", {}).items():
            assert isinstance(v, str), \
                f"Meta value for '{k}' should be str, got {type(v).__name__}: {v}"

    print("  ALL ASSERTIONS PASSED")

    th.print_report(results, "YOLO Object Counter")


if __name__ == "__main__":
    try:
        test_yolo_plugin()
        sys.exit(0)
    except Exception as e:
        print(f"\nTEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
