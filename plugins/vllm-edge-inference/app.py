"""
vLLM Edge Inference Plugin for Sage/Waggle
Runs a vision-language model (VLM) on edge nodes to generate
natural-language scene descriptions from camera images.

Uses vLLM for high-throughput inference with PagedAttention.
Default model: Qwen3-VL-32B-Instruct (32B params, BF16, ~67 GB)
designed for nodes with 128 GB unified memory (e.g. Thor / DGX Spark).
Smaller models (Qwen3-VL-8B, Phi-3.5-vision) available via --model flag.

Pipeline:
  1. Capture frame from camera
  2. Send image + prompt to local vLLM OpenAI-compatible endpoint
  3. Publish scene description as waggle measurement
  4. Optionally upload the source image

Measurement topics:
  env.scene.description    — natural-language scene description
  env.scene.summary        — one-line summary
  upload                   — source camera image
"""
import argparse
import base64
import io
import json
import logging
import os
import tempfile
import time

import cv2
import numpy as np
import requests
from PIL import Image

from waggle.plugin import Plugin
from waggle.data.vision import Camera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vllm-edge-inference")


# ── vLLM client ─────────────────────────────────────────────────────
class VLLMClient:
    """Client for vLLM's OpenAI-compatible API running locally."""

    def __init__(self, base_url: str, model: str, max_tokens: int = 512,
                 temperature: float = 0.3, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.endpoint = f"{self.base_url}/v1/chat/completions"

    def health_check(self) -> bool:
        """Check if vLLM server is ready."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def wait_for_ready(self, max_wait: int = 600, poll_interval: int = 10):
        """Block until vLLM server is healthy."""
        logger.info("Waiting for vLLM server at %s ...", self.base_url)
        elapsed = 0
        while elapsed < max_wait:
            if self.health_check():
                logger.info("vLLM server ready after %ds", elapsed)
                return True
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(f"vLLM server not ready after {max_wait}s")

    def describe_image(self, pil_image: Image.Image, prompt: str) -> str:
        """Send an image + prompt to vLLM and return the text response."""
        # Encode image as base64 data URL
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        return data["choices"][0]["message"]["content"].strip()


# ── vLLM server launcher ────────────────────────────────────────────
def launch_vllm_server(model: str, port: int, gpu_mem_frac: float,
                       max_model_len: int, dtype: str,
                       enforce_eager: bool = False) -> None:
    """Launch vLLM server as a subprocess (sidecar pattern)."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_frac),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    logger.info("Launching vLLM: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# ── main loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="vLLM Edge Inference — scene description from cameras"
    )
    # Camera
    parser.add_argument("--stream", default="bottom_camera",
                        help="Camera stream name or RTSP URL")
    parser.add_argument("--interval", type=int, default=120,
                        help="Seconds between captures")
    parser.add_argument("--continuous", default="Y",
                        help="Y = loop, N = single-shot")

    # vLLM server
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct",
                        help="HuggingFace VLM model ID")
    parser.add_argument("--vllm-port", type=int, default=8000,
                        help="Port for the local vLLM server")
    parser.add_argument("--vllm-url", default="",
                        help="Override vLLM base URL (skip auto-launch)")
    parser.add_argument("--gpu-mem-frac", type=float, default=0.58,
                        help="GPU memory fraction for vLLM (0.58 for unified memory, 0.85 for discrete)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum context length")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Model dtype: half, float16, bfloat16, auto")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens in LLM response")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")

    parser.add_argument("--enforce-eager", action="store_true",
                        help="Skip CUDA graph capture (saves GPU memory on unified memory systems)")

    # Prompting
    parser.add_argument("--prompt", default=(
        "Describe this outdoor scene in detail. Include: "
        "weather/sky conditions, visible objects and their approximate counts, "
        "any people or animals, notable features, and time of day estimate. "
        "Be factual and concise — 2-3 sentences."
    ), help="VLM prompt for scene description")
    parser.add_argument("--summary-prompt", default=(
        "Provide a one-line summary (under 15 words) of what is happening "
        "in this image."
    ), help="VLM prompt for short summary")

    # Output
    parser.add_argument("--upload-image", default="Y",
                        help="Y = upload source image each cycle")

    args = parser.parse_args()

    # Determine vLLM URL — either user-provided or auto-launch
    vllm_proc = None
    if args.vllm_url:
        base_url = args.vllm_url
    else:
        base_url = f"http://localhost:{args.vllm_port}"
        vllm_proc = launch_vllm_server(
            model=args.model,
            port=args.vllm_port,
            gpu_mem_frac=args.gpu_mem_frac,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
        )

    client = VLLMClient(
        base_url=base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Wait for vLLM to load the model
    client.wait_for_ready(max_wait=600)

    camera = Camera(args.stream)

    try:
        with Plugin() as plugin:
            logger.info("Plugin started — stream=%s, model=%s, interval=%ds",
                         args.stream, args.model, args.interval)
            while True:
                try:
                    sample = camera.snapshot()
                    frame = sample.data  # numpy BGR

                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Get detailed description
                    description = client.describe_image(pil_image, args.prompt)
                    plugin.publish(
                        "env.scene.description",
                        description,
                        timestamp=sample.timestamp,
                        meta={"camera": args.stream, "model": args.model},
                    )
                    logger.info("Description: %s", description[:120])

                    # Get short summary
                    summary = client.describe_image(pil_image, args.summary_prompt)
                    plugin.publish(
                        "env.scene.summary",
                        summary,
                        timestamp=sample.timestamp,
                        meta={"camera": args.stream, "model": args.model},
                    )
                    logger.info("Summary: %s", summary)

                    # Upload source image
                    if args.upload_image == "Y":
                        stem = os.path.splitext(os.path.basename(args.stream))[0]
                        tmp_path = os.path.join(tempfile.gettempdir(),
                                                f"{stem}-described.jpg")
                        cv2.imwrite(tmp_path, frame)
                        plugin.upload_file(tmp_path, timestamp=sample.timestamp,
                                           meta={"camera": args.stream,
                                                 "description": description[:200]})
                        os.unlink(tmp_path)

                except requests.exceptions.ConnectionError:
                    logger.warning("vLLM server connection lost — retrying in 30s")
                    time.sleep(30)
                    continue
                except Exception:
                    logger.exception("Inference error")

                if args.continuous != "Y":
                    break
                time.sleep(args.interval)
    finally:
        if vllm_proc is not None:
            logger.info("Terminating vLLM server (pid=%d)", vllm_proc.pid)
            vllm_proc.terminate()
            vllm_proc.wait(timeout=30)


if __name__ == "__main__":
    main()
