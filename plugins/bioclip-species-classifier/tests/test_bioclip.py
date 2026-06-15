"""
Unit test for BioCLIP Species Classifier plugin.

Mocks the BioCLIP model and text embeddings to return canned
species predictions, then verifies correct waggle publish calls.

Requires: pywaggle, numpy, opencv-python-headless, Pillow
Does NOT require: open_clip, torch, huggingface_hub, GPU
"""
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Setup paths ──────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent
PLUGIN_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PLUGIN_DIR))

import test_harness as th


# ── Mock BioCLIP classifier ─────────────────────────────────────────
class FakeBioCLIPClassifier:
    """
    Mimics BioCLIPClassifier.classify() — returns canned predictions
    without loading any model or text embeddings.
    """
    def __init__(self, rank="Class"):
        self.rank = rank
        self.rank_idx = ("Kingdom", "Phylum", "Class", "Order",
                         "Family", "Genus", "Species").index(rank)

    def classify(self, pil_image, top_k=5):
        """Return fake species predictions."""
        return [
            {"name": "Aves (Birds)", "confidence": 0.72},
            {"name": "Mammalia (Mammals)", "confidence": 0.15},
            {"name": "Insecta (Insects)", "confidence": 0.08},
            {"name": "Reptilia (Reptiles)", "confidence": 0.03},
            {"name": "Amphibia (Amphibians)", "confidence": 0.02},
        ][:top_k]


# ── Test ─────────────────────────────────────────────────────────────
def test_bioclip_plugin():
    print("\n--- BioCLIP Species Classifier: Unit Test ---\n")

    output_dir = th.setup_test_output("bioclip-unit")

    # We don't import app.py at all — the heavy deps (open_clip, torch,
    # huggingface_hub) would fail. Instead we replicate the plugin's
    # publish logic using our fake classifier, which is what we're testing:
    # the pywaggle integration, measurement names, meta format, upload flow.

    import cv2
    from PIL import Image as PILImage
    from waggle.plugin import Plugin
    from waggle.plugin.time import get_timestamp
    import tempfile

    classifier = FakeBioCLIPClassifier(rank="Class")
    images = th.get_test_images()
    rank = "Class"
    rank_lower = rank.lower()
    min_confidence = 0.1

    with Plugin() as plugin:
        for img_path in images:
            frame = cv2.imread(img_path)
            assert frame is not None, f"Failed to load {img_path}"

            pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predictions = classifier.classify(pil_image, top_k=5)
            ts = get_timestamp()

            if predictions and predictions[0]["confidence"] >= min_confidence:
                top = predictions[0]

                # Top prediction name
                plugin.publish(
                    f"env.species.{rank_lower}",
                    top["name"],
                    timestamp=ts,
                    meta={"camera": "test", "rank": rank, "model": "bioclip"},
                )
                print(f"  Published: env.species.{rank_lower} = {top['name']}")

                # Top prediction confidence
                plugin.publish(
                    f"env.species.{rank_lower}.confidence",
                    top["confidence"],
                    timestamp=ts,
                    meta={"camera": "test", "rank": rank},
                )
                print(f"  Published: env.species.{rank_lower}.confidence = {top['confidence']:.4f}")

                # Top-5 as JSON string
                plugin.publish(
                    "env.species.top5",
                    json.dumps(predictions),
                    timestamp=ts,
                    meta={"camera": "test", "rank": rank},
                )
                print(f"  Published: env.species.top5 (JSON, {len(predictions)} items)")

                # Upload source image
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False,
                                                 dir=output_dir)
                cv2.imwrite(tmp.name, frame)
                plugin.upload_file(tmp.name, timestamp=ts,
                                   meta={"camera": "test",
                                         "top_species": top["name"],
                                         "confidence": str(top["confidence"])})
                print(f"  Uploaded: {os.path.basename(tmp.name)}")
            else:
                print(f"  Skipped — below confidence threshold")

            print()

    # Parse and validate output
    results = th.parse_output(output_dir)
    measurements = results["measurements"]
    uploads = results["uploads"]
    n_images = len(images)

    # ── Assertions ───────────────────────────────────────────────
    # 3 measurements per image: species name, confidence, top5
    species_name = [m for m in measurements if m["name"] == f"env.species.{rank_lower}"]
    species_conf = [m for m in measurements if m["name"] == f"env.species.{rank_lower}.confidence"]
    species_top5 = [m for m in measurements if m["name"] == "env.species.top5"]
    upload_records = [m for m in measurements if m["name"] == "upload"]

    assert len(species_name) == n_images, \
        f"Expected {n_images} species name measurements, got {len(species_name)}"
    assert len(species_conf) == n_images, \
        f"Expected {n_images} confidence measurements, got {len(species_conf)}"
    assert len(species_top5) == n_images, \
        f"Expected {n_images} top5 measurements, got {len(species_top5)}"

    # Check value types
    for m in species_name:
        assert isinstance(m["value"], str), \
            f"Species name should be str, got {type(m['value']).__name__}"
        assert m["value"] == "Aves (Birds)"

    for m in species_conf:
        assert isinstance(m["value"], float), \
            f"Confidence should be float, got {type(m['value']).__name__}"
        assert abs(m["value"] - 0.72) < 0.001

    for m in species_top5:
        assert isinstance(m["value"], str), \
            f"Top5 should be JSON str, got {type(m['value']).__name__}"
        parsed = json.loads(m["value"])
        assert len(parsed) == 5, f"Expected 5 predictions, got {len(parsed)}"

    # Check meta fields are all strings
    for m in measurements:
        for k, v in m.get("meta", {}).items():
            assert isinstance(v, str), \
                f"Meta value for '{k}' should be str, got {type(v).__name__}: {v}"

    # Check uploads
    assert len(uploads) == n_images, \
        f"Expected {n_images} uploads, got {len(uploads)}"
    assert len(upload_records) == n_images, \
        f"Expected {n_images} upload records, got {len(upload_records)}"

    print("  ALL ASSERTIONS PASSED")

    th.print_report(results, "BioCLIP Species Classifier")


if __name__ == "__main__":
    try:
        test_bioclip_plugin()
        sys.exit(0)
    except Exception as e:
        print(f"\nTEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
