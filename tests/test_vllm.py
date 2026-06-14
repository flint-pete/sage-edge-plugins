"""
Unit test for vLLM Edge Inference plugin.

Mocks the vLLM HTTP API to return canned scene descriptions,
then verifies correct waggle publish calls and image uploads.

Requires: pywaggle, numpy, opencv-python-headless, Pillow
Does NOT require: vllm, torch, requests (to real server), GPU
"""
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Setup paths ──────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent
PLUGIN_DIR = PROJECT_DIR / "plugins" / "vllm-edge-inference"
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(PLUGIN_DIR))

import test_harness as th


# ── Mock VLLMClient ──────────────────────────────────────────────────
class FakeVLLMClient:
    """
    Mimics VLLMClient.describe_image() — returns canned descriptions
    without any HTTP calls or model loading.
    """
    DESCRIPTIONS = {
        "detail": (
            "An urban street scene under clear skies during midday. "
            "Three vehicles are visible including two sedans and a delivery truck. "
            "Two pedestrians walk along the sidewalk near commercial buildings. "
            "Street lights and traffic signals are present at the intersection."
        ),
        "summary": "Urban street with vehicles and pedestrians at midday.",
    }

    def __init__(self, model="test-model"):
        self.model = model
        self._call_count = 0

    def health_check(self):
        return True

    def wait_for_ready(self, max_wait=600):
        return True

    def describe_image(self, pil_image, prompt):
        self._call_count += 1
        # Alternate between detail and summary based on prompt length
        if "one-line summary" in prompt.lower() or "under 15 words" in prompt.lower():
            return self.DESCRIPTIONS["summary"]
        return self.DESCRIPTIONS["detail"]


# ── Test ─────────────────────────────────────────────────────────────
def test_vllm_plugin():
    print("\n--- vLLM Edge Inference: Unit Test ---\n")

    output_dir = th.setup_test_output("vllm-unit")

    # Replicate the plugin's publish logic using our fake client.
    # This tests: pywaggle integration, measurement names, meta format,
    # upload flow, and value constraints (str values for descriptions).

    import cv2
    from PIL import Image as PILImage
    from waggle.plugin import Plugin
    from waggle.plugin.time import get_timestamp
    import tempfile

    client = FakeVLLMClient(model="Qwen/Qwen3-VL-32B-Instruct")
    images = th.get_sample_images()

    detail_prompt = (
        "Describe this outdoor scene in detail. Include: "
        "weather/sky conditions, visible objects and their approximate counts, "
        "any people or animals, notable features, and time of day estimate. "
        "Be factual and concise — 2-3 sentences."
    )
    summary_prompt = (
        "Provide a one-line summary (under 15 words) of what is happening "
        "in this image."
    )

    with Plugin() as plugin:
        for img_path in images:
            frame = cv2.imread(img_path)
            assert frame is not None, f"Failed to load {img_path}"

            pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = get_timestamp()

            # Get detailed description
            description = client.describe_image(pil_image, detail_prompt)
            plugin.publish(
                "env.scene.description",
                description,
                timestamp=ts,
                meta={"camera": "test", "model": client.model},
            )
            print(f"  Published: env.scene.description ({len(description)} chars)")

            # Get short summary
            summary = client.describe_image(pil_image, summary_prompt)
            plugin.publish(
                "env.scene.summary",
                summary,
                timestamp=ts,
                meta={"camera": "test", "model": client.model},
            )
            print(f"  Published: env.scene.summary = {summary}")

            # Upload source image
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False,
                                             dir=output_dir)
            cv2.imwrite(tmp.name, frame)
            plugin.upload_file(tmp.name, timestamp=ts,
                               meta={"camera": "test",
                                     "description": description[:200]})
            print(f"  Uploaded: {os.path.basename(tmp.name)}")
            print()

    # Parse and validate output
    results = th.parse_output(output_dir)
    measurements = results["measurements"]
    uploads = results["uploads"]
    n_images = len(images)

    # ── Assertions ───────────────────────────────────────────────
    desc_measurements = [m for m in measurements if m["name"] == "env.scene.description"]
    summ_measurements = [m for m in measurements if m["name"] == "env.scene.summary"]
    upload_records = [m for m in measurements if m["name"] == "upload"]

    assert len(desc_measurements) == n_images, \
        f"Expected {n_images} descriptions, got {len(desc_measurements)}"
    assert len(summ_measurements) == n_images, \
        f"Expected {n_images} summaries, got {len(summ_measurements)}"

    # Check value types — all should be strings
    for m in desc_measurements:
        assert isinstance(m["value"], str), \
            f"Description should be str, got {type(m['value']).__name__}"
        assert len(m["value"]) > 50, \
            f"Description too short: {len(m['value'])} chars"

    for m in summ_measurements:
        assert isinstance(m["value"], str), \
            f"Summary should be str, got {type(m['value']).__name__}"
        assert len(m["value"]) < 200, \
            f"Summary too long: {len(m['value'])} chars"

    # Check meta fields are all strings
    for m in measurements:
        for k, v in m.get("meta", {}).items():
            assert isinstance(v, str), \
                f"Meta value for '{k}' should be str, got {type(v).__name__}: {v}"

    # Check that model is in meta
    for m in desc_measurements + summ_measurements:
        assert "model" in m.get("meta", {}), \
            f"Missing 'model' in meta for {m['name']}"

    # Check uploads
    assert len(uploads) == n_images, \
        f"Expected {n_images} uploads, got {len(uploads)}"
    assert len(upload_records) == n_images, \
        f"Expected {n_images} upload records, got {len(upload_records)}"

    # Verify the VLLMClient was called correctly
    assert client._call_count == 2 * n_images, \
        f"Expected {2*n_images} VLM calls, got {client._call_count}"

    print("  ALL ASSERTIONS PASSED")

    th.print_report(results, "vLLM Edge Inference")
    return True


if __name__ == "__main__":
    try:
        ok = test_vllm_plugin()
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"\nTEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
