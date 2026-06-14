# Sage Edge AI Plugins

Three standalone ML plugins for the [Sage Continuum](https://sagecontinuum.org)
edge computing platform. Each plugin captures frames from node cameras, runs
on-device inference using the node's GPU, and publishes compact measurements
via the Waggle data pipeline.

**Author:** Pete Beckman <pete.beckman@northwestern.edu>
**Created:** June 2026
**Platform:** Sage/Waggle (pywaggle SDK)
**Target Hardware:** Sage Thor nodes / NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory)

---

## Performance Summary

All three plugins tested end-to-end on NVIDIA DGX Spark (GB10 Blackwell, aarch64,
128GB unified memory, CUDA 13.0). Unit tests (mocked models) and integration tests
(real GPU inference) all pass.

| Plugin | Model | Model Size | Load Time | Inference | Notes |
|--------|-------|-----------|-----------|-----------|-------|
| YOLO Object Counter | yolo11x.pt | ~114M params | 2.7s | **20.6ms** avg | Largest YOLO11 variant |
| BioCLIP Species | BioCLIP-2 (pybioclip) | ViT-B/16 | 3.1s | **483ms** species, 2.9s class | 454K+ species, zero-shot |
| vLLM Scene Description | Qwen3-VL-32B-Instruct | 32B, BF16 (~63GB) | 405s | **26.3s** description (~97 tok) | Full 32B VLM, no quantization |

### Test Results

```
Unit tests:      3/3 passed (mocked models, no GPU required)
Integration tests:
  YOLO:          PASSED — 3 images, avg 20.6ms inference, 5 detections (birds)
  BioCLIP:       PASSED — 3 images, Class + Species rank, 9 measurements published
  vLLM:          PASSED — 3 images, ~97 token descriptions, 5.5s summaries
```

---

## Plugins

### 1. YOLO Object Counter (`plugins/yolo-object-counter/`)

Runs YOLO11x object detection on camera frames. Publishes per-class
object counts and uploads annotated images with bounding boxes.

| File | Purpose |
|------|---------|
| `app.py` | Main plugin — YOLODetector class, draw_boxes, publish loop (175 lines) |
| `Dockerfile` | nvcr.io/nvidia/pytorch:24.06-py3 base, pre-downloads yolo11x.pt |
| `sage.yaml` | ECR metadata (8 configurable inputs) |
| `requirements.txt` | pywaggle, ultralytics, torch, opencv, numpy, pillow |
| `ecr-meta/ecr-science-description.md` | Science justification for ECR |

**Model:** `yolo11x.pt` — the largest YOLO11 variant (114M params, 194 FLOPs).
Chosen over yolo11n for Thor nodes that have ample GPU memory. Loads in 2.7s,
warms up in 0.74s, and runs inference in ~20ms per 640x480 frame.

**Published measurements:**
- `env.count.<class_name>` — integer count per detected class (person, car, bird, etc.)
- `env.count.total` — total detections across all classes
- Annotated JPEG uploaded via `plugin.upload_file()`

**Key arguments:**
```
--stream bottom_camera     # Camera or RTSP URL
--model yolo11x.pt         # Model: yolov8n, yolo11n/s/m/l/x
--interval 30              # Seconds between captures
--conf-thres 0.25          # Confidence threshold
--classes "person,car"     # Filter to specific classes (empty = all)
```


### 2. BioCLIP Species Classifier (`plugins/bioclip-species-classifier/`)

Classifies biological organisms at any taxonomic rank using BioCLIP-2, a CLIP
model fine-tuned on TreeOfLife-10M (454K+ species). Uses the `pybioclip`
library with pre-computed text embeddings for fast similarity search.

| File | Purpose |
|------|---------|
| `app.py` | Main plugin — BioCLIPClassifier class, rank aggregation, publish loop (202 lines) |
| `Dockerfile` | Pre-downloads imageomics/bioclip-2 model + species text embeddings |
| `sage.yaml` | ECR metadata (8 configurable inputs) |
| `requirements.txt` | pywaggle, pybioclip, torch, pillow |
| `ecr-meta/ecr-science-description.md` | Science justification for ECR |

**Model:** `BioCLIP-2` via the `pybioclip` library (ViT-B/16 backbone). Upgraded
from the original BioCLIP to BioCLIP-2 for better accuracy. Loads in 3.1s,
species-level inference ~483ms, class-level ~2.9s per image.

**Published measurements:**
- `env.species.<rank>` — top predicted taxon name (e.g., "Aves", "Mammalia")
- `env.species.<rank>.confidence` — confidence score (0-1)
- `env.species.top5` — JSON array of top-5 predictions
- Source JPEG uploaded via `plugin.upload_file()`

**Key arguments:**
```
--stream bottom_camera     # Camera or RTSP URL
--rank Class               # Kingdom/Phylum/Class/Order/Family/Genus/Species
--interval 60              # Seconds between captures
--min-confidence 0.1       # Minimum confidence to publish
--top-k 5                  # Number of top predictions
```


### 3. vLLM Edge Inference (`plugins/vllm-edge-inference/`)

Runs vision-language models (VLMs) on edge nodes to generate natural-language
scene descriptions from camera images. Uses a sidecar architecture: launches
a local vLLM server, then communicates via the OpenAI-compatible API.

| File | Purpose |
|------|---------|
| `app.py` | Main plugin — VLLMClient class, sidecar launcher, publish loop (271 lines) |
| `Dockerfile` | Pre-downloads Qwen3-VL-32B-Instruct (~63GB BF16) |
| `sage.yaml` | ECR metadata (14 configurable inputs) |
| `requirements.txt` | pywaggle, vllm, torch, transformers, requests |
| `ecr-meta/ecr-science-description.md` | Science justification for ECR |

**Model:** `Qwen/Qwen3-VL-32B-Instruct` — full 32B parameter VLM in BF16
(~63GB VRAM). Chosen as the largest high-quality open VLM that fits in 128GB
unified memory with room for KV cache. Generates detailed scene descriptions
averaging 97 tokens in ~26s, and one-line summaries in ~5.5s.

**Published measurements:**
- `env.scene.description` — detailed natural-language scene description
- `env.scene.summary` — one-line summary (under 15 words)
- Source JPEG uploaded via `plugin.upload_file()`

**Key arguments:**
```
--stream bottom_camera                       # Camera or RTSP URL
--model Qwen/Qwen3-VL-32B-Instruct          # Any HF VLM supported by vLLM
--interval 120                               # Seconds between captures
--gpu-mem-frac 0.58                          # GPU memory fraction (see below)
--enforce-eager                              # Skip CUDA graph capture
--prompt "Describe this scene..."            # Custom prompt
--max-model-len 4096                         # Context length
--dtype bfloat16                             # Model precision
```


## Unified Memory Notes (DGX Spark / Thor Nodes)

The NVIDIA GB10 (Blackwell) in DGX Spark and Sage Thor nodes uses **unified
memory** — CPU and GPU share the same 128GB pool. This creates important
differences from discrete-GPU systems:

1. **gpu-mem-frac must be ~0.58, not 0.85.** vLLM reports ~121.7 GiB total
   GPU memory, but 45-75 GiB is consumed by the OS and system processes.
   Setting 0.85 requests ~103 GiB and OOMs. 0.58 requests ~70.6 GiB and
   fits in the ~73 GiB typically available.

2. **--enforce-eager is required.** CUDA graph capture allocates ~5 GB for
   profiling runs. On a 32B model using ~63 GiB, this pushes past available
   memory and crashes during KV cache initialization. Eager mode trades a
   small throughput penalty for reliable startup.

3. **Orphaned processes are catastrophic.** If a vLLM EngineCore process
   survives after server shutdown, it holds GPU memory. There's no discrete
   GPU to `nvidia-smi` reset — you must `kill` the orphan. Always verify
   cleanup after test failures.

4. **Server startup takes ~7-17 minutes.** The 32B BF16 model loads in ~400s,
   then Triton JIT compilation and warmup add another 2-3 minutes.


## Job Specifications (`plugins/jobs/`)

| File | Description |
|------|-------------|
| `yolo-counter-job.yaml` | YOLO counting people/cars on node W097 (Chicago) |
| `bioclip-species-job.yaml` | BioCLIP at Order rank on node W019 |
| `vllm-scene-job.yaml` | vLLM scene description on node W097 |
| `combined-ml-pipeline-job.yaml` | All three plugins running together on one node |


## Local Testing

Plugins can be tested locally without Sage credentials using pywaggle's
local logging mode. Instead of publishing to the Sage data pipeline, all
measurements and file uploads are captured to local files:

```bash
export PYWAGGLE_LOG_DIR=./test-output
python3 app.py --stream test-image.jpg --continuous N

# Inspect results:
cat test-output/data.ndjson      # Published measurements (one JSON per line)
ls test-output/uploads/          # Uploaded image files
```

### Test Suite

The `tests/` directory contains a complete test harness with two levels:

**Unit Tests** (no GPU required — mocked models):
```bash
cd tests
pip install -r requirements.txt
python -m pytest test_yolo.py test_bioclip.py test_vllm.py -v
```

**Integration Tests** (real GPU inference — requires CUDA device):
```bash
cd tests
python test_yolo_integration.py      # ~30s: loads yolo11x.pt, runs 3 images
python test_bioclip_integration.py   # ~60s: loads BioCLIP-2, Class + Species
python test_vllm_integration.py      # ~25min: downloads/loads Qwen3-VL-32B
```

Test output goes to `tests/output/{yolo,bioclip,vllm}-integration/` with
`test_results.json` (full metrics) and `data.ndjson` + `uploads/` (published data).


## Architecture Notes

- All plugins use the **pywaggle SDK** (`waggle.plugin.Plugin`, `waggle.data.vision.Camera`)
- Camera abstraction accepts: named streams (`bottom_camera`), RTSP URLs, or
  local image paths (for testing)
- Dockerfiles use `nvcr.io/nvidia/pytorch:24.06-py3` base (CUDA + PyTorch)
- Models are pre-downloaded at build time (edge nodes may lack internet)
- Supports ARM64 (Thor nodes, DGX Spark) and AMD64 (Sage Blades)
- vLLM plugin uses sidecar pattern: separate server process + OpenAI API client

## Plugin Tutorials

Each plugin includes a detailed `overview.md` tutorial designed for training
students on how the code works, how to test locally, and how to deploy:

| Plugin | Tutorial | Size | Highlights |
|--------|----------|------|------------|
| YOLO Object Counter | [overview.md](plugins/yolo-object-counter/overview.md) | 15 KB | YOLODetector internals, draw_boxes, model selection guide |
| BioCLIP Species | [overview.md](plugins/bioclip-species-classifier/overview.md) | 16 KB | Taxonomic ranks explained, pybioclip abstraction, accuracy tradeoffs |
| vLLM Edge Inference | [overview.md](plugins/vllm-edge-inference/overview.md) | 25 KB | Sidecar architecture, GPU memory tuning, unified memory pitfalls |


## File Inventory

```
plugins/
  yolo-object-counter/
    app.py                              # 175 lines — YOLO detection + counting
    Dockerfile                          # NVIDIA PyTorch base, bakes yolo11x.pt
    requirements.txt                    # 6 dependencies
    sage.yaml                           # ECR metadata, 8 inputs with descriptions
    overview.md                         # 15KB tutorial — internals, testing, deployment
    ecr-meta/
      README                            # ECR submission instructions
      ecr-science-description.md        # Science narrative for ECR portal
      ecr-credits-license.txt           # Authors, funding (NSF 1935984), license
      ecr-project-keywords.txt          # Search keywords (ontology-aligned)
      ecr-project-url.txt               # GitHub URL

  bioclip-species-classifier/
    app.py                              # 202 lines — BioCLIP-2 classification
    Dockerfile                          # Bakes BioCLIP-2 model + text embeddings
    requirements.txt                    # pybioclip + dependencies
    sage.yaml                           # ECR metadata, 8 inputs with descriptions
    overview.md                         # 16KB tutorial — taxonomy, testing, deployment
    ecr-meta/
      README                            # ECR submission instructions
      ecr-science-description.md        # Science narrative for ECR portal
      ecr-credits-license.txt           # Authors, funding, license
      ecr-project-keywords.txt          # Search keywords
      ecr-project-url.txt               # GitHub URL

  vllm-edge-inference/
    app.py                              # 271 lines — vLLM sidecar + VLM inference
    Dockerfile                          # Bakes Qwen3-VL-32B-Instruct (~63GB)
    requirements.txt                    # 9 dependencies
    sage.yaml                           # ECR metadata, 14 inputs with descriptions
    overview.md                         # 25KB tutorial — sidecar arch, memory tuning
    ecr-meta/
      README                            # ECR submission instructions
      ecr-science-description.md        # Science narrative for ECR portal
      ecr-credits-license.txt           # Authors, funding, license
      ecr-project-keywords.txt          # Search keywords
      ecr-project-url.txt               # GitHub URL

  jobs/
    yolo-counter-job.yaml               # YOLO on W097 (Chicago)
    bioclip-species-job.yaml            # BioCLIP on W019
    vllm-scene-job.yaml                 # vLLM on W097
    combined-ml-pipeline-job.yaml       # All 3 plugins on one node

tests/
    generate_test_images.py             # 190 lines — synthetic test image generator
    test_yolo.py                        # 170 lines — unit tests (mocked)
    test_yolo_integration.py            # 248 lines — GPU integration test
    test_bioclip.py                     # 188 lines — unit tests (mocked)
    test_bioclip_integration.py         # 240 lines — GPU integration test
    test_vllm.py                        # 193 lines — unit tests (mocked)
    test_vllm_integration.py            # 343 lines — GPU integration test
    test_harness.py                     # 152 lines — shared test utilities
```

### ECR Submission Readiness

Each plugin is structured for Sage ECR (Edge Code Repository) submission.
Before submitting, two image files must still be created in each ecr-meta/:
- `ecr-icon.jpg` — 512x512 plugin icon
- `ecr-science-image.jpg` — 1920x1080+ science image for the portal

See `ecr-meta/README` in each plugin for step-by-step submission instructions.


## Research Notes

Detailed research on the Sage platform, APIs, node architecture, and
development patterns is in `sage-agents.md`.
