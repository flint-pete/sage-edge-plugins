#!/usr/bin/env python3
"""
vLLM Edge Inference INTEGRATION TEST — Real Model Inference

Launches a real vLLM server with Qwen3-VL-32B-Instruct, sends
test images for scene description, and publishes results via pywaggle.

Requirements:
  - GPU with sufficient VRAM (tested on 128GB unified memory)
  - vllm, torch, pywaggle, Pillow, requests installed
  - ~67 GB disk for model weights (auto-downloaded from HuggingFace)

Usage:
  python test_vllm_integration.py
"""
import base64
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import requests
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-VL-32B-Instruct"
VLLM_PORT = 8199  # Non-default port to avoid conflicts
VLLM_URL = f"http://localhost:{VLLM_PORT}"
MAX_MODEL_LEN = 4096
GPU_MEM_FRAC = 0.58  # Unified memory: ~73GiB free of 122GiB reported by vLLM
DTYPE = "bfloat16"
MAX_TOKENS = 512
TEMPERATURE = 0.3
SERVER_TIMEOUT = 2400  # 40 min for first-time model download (~67GB) + load

DESCRIPTION_PROMPT = (
    "Describe this outdoor scene in detail. Include: "
    "weather/sky conditions, visible objects and their approximate counts, "
    "any people or animals, notable features, and time of day estimate. "
    "Be factual and concise — 2-3 sentences."
)
SUMMARY_PROMPT = (
    "Provide a one-line summary (under 15 words) of what is happening "
    "in this image."
)


def get_test_images():
    """Get test images from the test-images directory."""
    img_dir = os.path.join(os.path.dirname(__file__), "test-images")
    images = {}
    if not os.path.isdir(img_dir):
        return images
    for name in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            path = os.path.join(img_dir, name)
            if os.path.isfile(path):
                images[name] = Image.open(path).convert("RGB")
    return images


def launch_vllm_server():
    """Launch vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", str(GPU_MEM_FRAC),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--dtype", DTYPE,
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--enforce-eager",  # Skip CUDA graphs to save GPU memory
    ]
    print(f"  Launching: {' '.join(cmd)}")
    # Write server logs to a file for debugging
    log_path = os.path.join(
        os.path.dirname(__file__), "output", "vllm-integration", "server.log"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    server_log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(timeout=SERVER_TIMEOUT, poll=5):
    """Poll the vLLM health endpoint until ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{VLLM_URL}/health", timeout=3)
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"  Server ready after {elapsed:.1f}s")
                return elapsed
        except requests.ConnectionError:
            pass
        time.sleep(poll)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


def describe_image(pil_image, prompt):
    """Send image + prompt to vLLM and return response text + timing."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    t0 = time.time()
    r = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    elapsed_ms = (time.time() - t0) * 1000
    r.raise_for_status()
    data = r.json()

    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    return text, elapsed_ms, usage


def main():
    print("=" * 70)
    print("vLLM INTEGRATION TEST — Real Model Inference")
    print(f"Model: {MODEL}")
    print(f"Dtype: {DTYPE}")
    print(f"Max model len: {MAX_MODEL_LEN}")
    print(f"GPU memory fraction: {GPU_MEM_FRAC}")
    print("=" * 70)

    # Load test images
    images = get_test_images()
    if not images:
        print("ERROR: No test images found in test-images/")
        sys.exit(1)
    print(f"\nFound {len(images)} test images: {', '.join(images.keys())}")

    # Set up output directory
    out_dir = os.path.join(os.path.dirname(__file__), "output", "vllm-integration")
    os.makedirs(out_dir, exist_ok=True)
    os.environ["PYWAGGLE_LOG_DIR"] = out_dir

    # Launch vLLM server
    print(f"\n[1/5] Launching vLLM server on port {VLLM_PORT}...")
    server_proc = launch_vllm_server()

    try:
        # Wait for server to be ready
        print("\n[2/5] Waiting for model load (this may download ~67GB on first run)...")
        load_time = wait_for_server()

        # Warmup inference
        print("\n[3/5] Warmup inference...")
        first_img = list(images.values())[0]
        warmup_text, warmup_ms, _ = describe_image(first_img, "What is in this image?")
        print(f"  Warmup done in {warmup_ms:.0f}ms")
        print(f"  Response: {warmup_text[:100]}...")

        # Run inference on all test images
        print("\n[4/5] Running inference on test images...")
        results = []

        for img_name, pil_img in images.items():
            w, h = pil_img.size
            print(f"\n  {img_name} ({w}x{h})")

            # Get detailed description
            desc_text, desc_ms, desc_usage = describe_image(pil_img, DESCRIPTION_PROMPT)
            print(f"    Description ({desc_ms:.0f}ms, "
                  f"{desc_usage.get('completion_tokens', '?')} tokens):")
            print(f"      {desc_text[:200]}")

            # Get summary
            sum_text, sum_ms, sum_usage = describe_image(pil_img, SUMMARY_PROMPT)
            print(f"    Summary ({sum_ms:.0f}ms, "
                  f"{sum_usage.get('completion_tokens', '?')} tokens):")
            print(f"      {sum_text}")

            results.append({
                "image": img_name,
                "size": f"{w}x{h}",
                "description": desc_text,
                "description_ms": desc_ms,
                "description_tokens": desc_usage.get("completion_tokens", 0),
                "summary": sum_text,
                "summary_ms": sum_ms,
                "summary_tokens": sum_usage.get("completion_tokens", 0),
            })

        # Publish results via pywaggle
        print("\n[5/5] Publishing results via pywaggle...")
        from waggle.plugin import Plugin

        with Plugin() as plugin:
            for r in results:
                ts = int(time.time_ns())

                plugin.publish(
                    "env.scene.description",
                    r["description"],
                    timestamp=ts,
                    meta={"camera": "test", "model": MODEL,
                          "image": r["image"],
                          "inference_ms": str(int(r["description_ms"])),
                          "tokens": str(r["description_tokens"])},
                )
                plugin.publish(
                    "env.scene.summary",
                    r["summary"],
                    timestamp=ts,
                    meta={"camera": "test", "model": MODEL,
                          "image": r["image"],
                          "inference_ms": str(int(r["summary_ms"])),
                          "tokens": str(r["summary_tokens"])},
                )

                # Upload the source image
                img_path = os.path.join(
                    os.path.dirname(__file__), "test-images", r["image"]
                )
                plugin.upload_file(
                    img_path, timestamp=ts,
                    meta={"camera": "test",
                          "description": r["description"][:200]},
                )

        # Verify output
        ndjson_path = os.path.join(out_dir, "data.ndjson")
        upload_dir = os.path.join(out_dir, "uploads")

        ndjson_lines = 0
        if os.path.exists(ndjson_path):
            with open(ndjson_path) as f:
                ndjson_lines = sum(1 for _ in f)

        upload_count = 0
        if os.path.exists(upload_dir):
            upload_count = len(os.listdir(upload_dir))

        # Compute averages
        avg_desc_ms = sum(r["description_ms"] for r in results) / len(results)
        avg_sum_ms = sum(r["summary_ms"] for r in results) / len(results)
        avg_desc_tok = sum(r["description_tokens"] for r in results) / len(results)

        # Save detailed results
        test_results = {
            "model": MODEL,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_mem_frac": GPU_MEM_FRAC,
            "server_load_time_s": round(load_time, 1),
            "warmup_ms": round(warmup_ms, 0),
            "images_processed": len(results),
            "avg_description_ms": round(avg_desc_ms, 0),
            "avg_summary_ms": round(avg_sum_ms, 0),
            "avg_description_tokens": round(avg_desc_tok, 0),
            "published_measurements": ndjson_lines,
            "uploaded_images": upload_count,
            "per_image_results": results,
        }

        results_path = os.path.join(out_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Model: {MODEL}")
        print(f"Dtype: {DTYPE}")
        print(f"Server load time: {load_time:.1f}s")
        print(f"Warmup time: {warmup_ms:.0f}ms")
        print(f"Images processed: {len(results)}")
        print(f"Avg description inference: {avg_desc_ms:.0f}ms ({avg_desc_tok:.0f} tokens)")
        print(f"Avg summary inference: {avg_sum_ms:.0f}ms")
        print(f"Published measurements: {ndjson_lines}")
        print(f"Uploaded images: {upload_count}")
        print(f"NDJSON records: {ndjson_lines}")
        print(f"Upload files: {upload_count}")

        # Assertions
        passed = True
        checks = [
            (len(results) == len(images), f"processed all {len(images)} images"),
            (ndjson_lines >= len(images) * 2, f"at least {len(images)*2} measurements"),
            (upload_count >= len(images), f"at least {len(images)} uploads"),
            (all(len(r["description"]) > 20 for r in results), "descriptions non-empty"),
            (all(len(r["summary"]) > 5 for r in results), "summaries non-empty"),
            (avg_desc_ms < 60000, "avg description under 60s"),
        ]

        for ok, label in checks:
            status = "✓" if ok else "✗"
            print(f"  {status} {label}")
            if not ok:
                passed = False

        print()
        if passed:
            print("✓ vLLM INTEGRATION TEST PASSED")
        else:
            print("✗ vLLM INTEGRATION TEST FAILED")
            sys.exit(1)

        print(f"\nDetailed results: {results_path}")

    finally:
        # Shutdown vLLM server
        print("\nShutting down vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
