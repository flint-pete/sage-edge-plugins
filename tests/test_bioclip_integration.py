#!/usr/bin/env python3
"""
BioCLIP2 Integration Test — Real Model Inference on DGX Spark
Downloads BioCLIP2 (imageomics/bioclip-2) + TreeOfLife-200M embeddings,
runs real species classification on test images, verifies pywaggle publish/upload.

Requires: GPU with CUDA, pybioclip, pywaggle, opencv, torch
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
from PIL import Image

from bioclip import Rank
from bioclip.predict import TreeOfLifeClassifier

# ── pywaggle local output setup ─────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "bioclip-integration")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["PYWAGGLE_LOG_DIR"] = OUTPUT_DIR

from waggle.plugin import Plugin

# ── paths ────────────────────────────────────────────────────────────
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample-images")
IMAGES = ["urban_street.jpg", "wildlife.jpg", "sky_clouds.jpg"]

MODEL_STR = "hf-hub:imageomics/bioclip-2"
RANKS_TO_TEST = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
RANK_MAP = {
    "Kingdom": Rank.KINGDOM, "Phylum": Rank.PHYLUM, "Class": Rank.CLASS,
    "Order": Rank.ORDER, "Family": Rank.FAMILY, "Genus": Rank.GENUS,
    "Species": Rank.SPECIES,
}


def main():
    print("=" * 70)
    print("BioCLIP2 INTEGRATION TEST — Real Model Inference")
    print(f"Model: {MODEL_STR}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── 1. Load model ────────────────────────────────────────────────
    print(f"\n[1/4] Loading BioCLIP2 TreeOfLifeClassifier...")
    t0 = time.time()
    classifier = TreeOfLifeClassifier(model_str=MODEL_STR)
    load_time = time.time() - t0
    print(f"  Model + embeddings loaded in {load_time:.2f}s")

    # ── 2. Warmup ────────────────────────────────────────────────────
    print("\n[2/4] Warmup inference...")
    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    t0 = time.time()
    classifier.predict([dummy], rank=Rank.CLASS, k=3)
    warmup_time = time.time() - t0
    print(f"  Warmup done in {warmup_time:.2f}s")

    # ── 3. Run inference on test images ──────────────────────────────
    print(f"\n[3/4] Running inference on {len(IMAGES)} test images...")
    all_results = []

    for img_name in IMAGES:
        img_path = os.path.join(SAMPLE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  SKIP {img_name} — not found")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  SKIP {img_name} — failed to read")
            continue

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]

        img_result = {
            "image": img_name,
            "resolution": f"{w}x{h}",
            "ranks": {},
        }

        # Test classification at Class rank (primary) and Species rank
        for rank_name in ["Class", "Species"]:
            t0 = time.time()
            predictions = classifier.predict(
                [pil_image], rank=RANK_MAP[rank_name], k=5
            )
            inf_time = time.time() - t0

            # pybioclip returns list of dicts with rank keys + 'score'
            rank_key = rank_name.lower()
            sorted_preds = []
            for r in predictions:
                if rank_key == "species":
                    name = r.get("species", r.get("genus", "Unknown"))
                else:
                    name = r.get(rank_key, "Unknown")
                sorted_preds.append({"name": name, "confidence": float(r["score"])})

            rank_result = {
                "inference_ms": round(inf_time * 1000, 1),
                "predictions": sorted_preds[:5],
            }
            img_result["ranks"][rank_name] = rank_result

            print(f"\n  {img_name} ({w}x{h}) — Rank: {rank_name}")
            print(f"    Inference: {inf_time*1000:.1f}ms")
            for i, pred in enumerate(sorted_preds[:5], 1):
                print(f"    #{i}: {pred['name']} ({pred['confidence']:.6f})")

        all_results.append(img_result)

    # ── 4. Publish via pywaggle ──────────────────────────────────────
    print(f"\n[4/4] Publishing results via pywaggle...")
    published = 0
    uploaded = 0

    with Plugin() as plugin:
        for result in all_results:
            img_name = result["image"]
            img_path = os.path.join(SAMPLE_DIR, img_name)
            frame = cv2.imread(img_path)
            ts = time.time_ns()

            # Use Class rank predictions for publishing
            class_preds = result["ranks"].get("Class", {}).get("predictions", [])
            if class_preds and class_preds[0]["confidence"] >= 0.01:
                top = class_preds[0]

                plugin.publish(
                    "env.species.class", top["name"],
                    timestamp=ts,
                    meta={"camera": "test", "rank": "Class", "model": MODEL_STR},
                )
                published += 1

                plugin.publish(
                    "env.species.class.confidence", top["confidence"],
                    timestamp=ts,
                    meta={"camera": "test", "rank": "Class"},
                )
                published += 1

                plugin.publish(
                    "env.species.top5",
                    json.dumps(class_preds),
                    timestamp=ts,
                    meta={"camera": "test", "rank": "Class"},
                )
                published += 1

                # Upload image
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=OUTPUT_DIR)
                cv2.imwrite(tmp.name, frame)
                plugin.upload_file(tmp.name, timestamp=ts,
                                   meta={"camera": "test",
                                         "top_species": top["name"],
                                         "confidence": str(top["confidence"])})
                uploaded += 1

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_STR}")
    print(f"Model load time: {load_time:.2f}s")
    print(f"Warmup time: {warmup_time:.2f}s")
    print(f"Images processed: {len(all_results)}")
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
    if not all_results:
        print("\nFAIL: No images processed!")
        passed = False
    if published == 0:
        print("\nFAIL: No measurements published!")
        passed = False
    if uploaded == 0:
        print("\nFAIL: No images uploaded!")
        passed = False

    # Verify all images got predictions
    for result in all_results:
        for rank_name, rank_data in result["ranks"].items():
            if not rank_data["predictions"]:
                print(f"\nFAIL: No predictions for {result['image']} at {rank_name}")
                passed = False

    if passed:
        print("\n✓ BioCLIP2 INTEGRATION TEST PASSED")
    else:
        print("\n✗ BioCLIP2 INTEGRATION TEST FAILED")
        sys.exit(1)

    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_STR,
            "load_time_s": round(load_time, 2),
            "warmup_time_s": round(warmup_time, 2),
            "images": all_results,
            "published": published,
            "uploaded": uploaded,
        }, f, indent=2)
    print(f"\nDetailed results: {results_path}")


if __name__ == "__main__":
    main()
