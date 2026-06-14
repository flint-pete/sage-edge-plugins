# vLLM Edge Inference — Plugin Overview

A comprehensive tutorial for understanding, configuring, and deploying the
vLLM Edge Inference plugin on the Sage Continuum edge computing platform.

---

## Table of Contents

1. [What This Plugin Does](#what-this-plugin-does)
2. [File Layout](#file-layout)
3. [How It Works (Internals)](#how-it-works-internals)
4. [Configuration Reference](#configuration-reference)
5. [Understanding the Sidecar Architecture](#understanding-the-sidecar-architecture)
6. [Local Testing (No Node Required)](#local-testing)
7. [Deployment on a Sage Node](#deployment-on-a-sage-node)
8. [Data Published](#data-published)
9. [Querying Your Data](#querying-your-data)
10. [Key Issues and Pitfalls](#key-issues-and-pitfalls)
11. [Example Scenarios](#example-scenarios)

---

## 1. What This Plugin Does

The vLLM Edge Inference plugin captures frames from a camera, sends each
image to a locally-running vision-language model (VLM), and publishes the
model's natural-language scene description to the Sage data store.

Think of it as an always-on narrator: every two minutes the node looks at
what the camera sees and writes "Overcast sky, light rain on wet pavement.
Three pedestrians with umbrellas near a crosswalk.  Mid-afternoon based on
diffuse lighting."

**What is vLLM?** vLLM is a high-throughput inference engine for large
language models.  It uses PagedAttention to manage GPU memory efficiently,
enabling models with tens of billions of parameters to run on a single GPU.
Its OpenAI-compatible API means the plugin communicates with it using
standard HTTP requests — the same format used by ChatGPT.

**Why a 32-billion parameter model at the edge?** Traditional edge CV
plugins (YOLO, BioCLIP) produce structured outputs — bounding boxes, class
labels.  A VLM produces free-form natural language: it can describe spatial
relationships ("a truck parked behind a fence"), infer context ("construction
site, possibly active"), and answer arbitrary questions about scenes.  The
Sage Thor nodes with 128 GB unified memory make this feasible without
cloud connectivity.

**Default model: Qwen3-VL-32B-Instruct** — a 32B-parameter vision-language
model from Alibaba, running in BF16 precision (~67 GB weights).  It
achieves state-of-the-art scores on visual question answering benchmarks
while fitting within the Thor/DGX Spark memory envelope.

---

## 2. File Layout

```
vllm-edge-inference/
├── app.py                          # Main application (271 lines)
├── Dockerfile                      # Container build — downloads 67GB model at build time
├── requirements.txt                # Python dependencies
├── sage.yaml                       # ECR metadata
├── overview.md                     # This file
├── ecr-meta/                       # ECR submission metadata
│   ├── README                      # Instructions for ECR submission
│   ├── ecr-science-description.md  # Science narrative
│   ├── ecr-credits-license.txt     # Authors, funding, license
│   ├── ecr-project-keywords.txt    # Search keywords
│   └── ecr-project-url.txt         # Project URL
│   (ecr-icon.jpg)                  # 512x512 icon — create before submission
│   (ecr-science-image.jpg)         # 1920x1080 science image — create before submission
├── jobs/                           # Sample Sage job specifications
│   └── vllm-scene-job.yaml        # Deploy on W097 with Qwen3-VL-32B
└── tests/                          # Self-contained test suite
    ├── run-tests.sh                # Run all tests for this plugin
    ├── test_vllm.py                # Unit test (mocked model, no GPU)
    ├── test_vllm_integration.py    # Integration test (real model, GPU)
    ├── test_harness.py             # Pywaggle test harness library
    └── sample-images/              # Synthetic images for unit tests
```

### What each file does

| File | Purpose |
|------|---------|
| `app.py` | VLLMClient class, server launcher, camera capture, publishing |
| `Dockerfile` | Based on nvidia/pytorch, installs vLLM, pre-downloads Qwen3-VL-32B |
| `requirements.txt` | pywaggle, vllm, requests, opencv-python-headless, pillow |
| `sage.yaml` | Plugin name, version, 14 configurable inputs with descriptions |
| `ecr-meta/` | Portal display metadata: science description, keywords, credits |

---

## 3. How It Works (Internals)

This plugin has a **two-process architecture**: the plugin process (app.py)
and the vLLM server process (sidecar).  They communicate over localhost HTTP.

### Startup Sequence

```
1. Parse command-line arguments (14 flags)
2. Determine vLLM URL:
   ├── If --vllm-url provided → use external server (skip launch)
   └── Else → launch_vllm_server() as a subprocess
       ├── Spawns: python -m vllm.entrypoints.openai.api_server
       │   --model Qwen/Qwen3-VL-32B-Instruct
       │   --port 8000
       │   --gpu-memory-utilization 0.58
       │   --max-model-len 4096
       │   --dtype bfloat16
       │   --enforce-eager
       │   --no-enable-log-requests
       └── Returns immediately (Popen, non-blocking)
3. Create VLLMClient(base_url, model, max_tokens, temperature)
4. client.wait_for_ready(max_wait=600)
   └── Polls GET /health every 10s until 200 OK
   └── Model loading takes ~7 minutes (405s measured, from cache)
5. Open camera stream via pywaggle Camera
6. Enter main loop
```

### Main Loop (one iteration)

```
┌──────────────────────────────────────────────────────────┐
│ 1. cam.snapshot()                                        │
│    └── Captures a single frame → ImageSample (BGR numpy) │
│    └── Convert BGR→RGB → PIL Image                       │
│                                                          │
│ 2. client.describe_image(image, prompt)                  │
│    ├── Encode PIL image → JPEG → base64 data URL         │
│    ├── POST /v1/chat/completions                         │
│    │   {"model": "Qwen/Qwen3-VL-32B-Instruct",          │
│    │    "messages": [{"role":"user","content":[           │
│    │      {"type":"image_url","image_url":{"url":"..."}}, │
│    │      {"type":"text","text":"Describe this scene..."}]}│
│    │    ], "max_tokens": 512, "temperature": 0.3}        │
│    └── Returns: "Overcast sky, wet pavement. Three..."   │
│                                                          │
│ 3. plugin.publish("env.scene.description", description)  │
│                                                          │
│ 4. client.describe_image(image, summary_prompt)          │
│    └── Returns: "Rainy urban crosswalk with pedestrians" │
│                                                          │
│ 5. plugin.publish("env.scene.summary", summary)          │
│                                                          │
│ 6. Upload source image (JPEG) if --upload-image=Y       │
│    └── plugin.upload_file(tmp.jpg, meta={description})   │
│                                                          │
│ 7. Sleep(interval) and repeat (if continuous=Y)          │
└──────────────────────────────────────────────────────────┘
```

### Key Classes

**`VLLMClient`** (lines 46–111)
- `__init__(base_url, model, max_tokens, temperature, timeout)`: configures
  the HTTP endpoint
- `health_check()`: GET /health, returns True/False
- `wait_for_ready(max_wait, poll_interval)`: blocks until server is healthy
- `describe_image(pil_image, prompt)`: encodes image as base64, sends
  OpenAI-format chat completion request, returns text response

**`launch_vllm_server()`** (lines 115–139)
- Spawns vLLM's OpenAI-compatible server via `subprocess.Popen`
- Constructs the full CLI command from function arguments
- Routes stdout to DEVNULL, stderr to PIPE
- Returns immediately — the caller must `wait_for_ready()`

### The OpenAI-Compatible API

The plugin uses the same API format as OpenAI's GPT-4V:

```json
{
  "model": "Qwen/Qwen3-VL-32B-Instruct",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "Describe this scene..."}
    ]
  }],
  "max_tokens": 512,
  "temperature": 0.3
}
```

This means you can swap models by changing `--model` without touching
any code — any VLM that vLLM supports and that fits in memory will work.

---

## 4. Configuration Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--stream` | string | `bottom_camera` | Camera source name or RTSP URL |
| `--interval` | int | `120` | Seconds between captures |
| `--continuous` | string | `Y` | `Y` = loop, `N` = single-shot |
| `--model` | string | `Qwen/Qwen3-VL-32B-Instruct` | HuggingFace VLM model ID |
| `--vllm-port` | int | `8000` | Port for auto-launched vLLM server |
| `--vllm-url` | string | (empty) | Override URL — skips auto-launch |
| `--gpu-mem-frac` | float | `0.58` | GPU memory fraction for vLLM |
| `--max-model-len` | int | `4096` | Maximum context length (tokens) |
| `--dtype` | string | `bfloat16` | Model precision: bfloat16, float16, half, auto |
| `--max-tokens` | int | `512` | Max tokens in each LLM response |
| `--temperature` | float | `0.3` | Sampling temperature (0=deterministic) |
| `--enforce-eager` | flag | off | Skip CUDA graph capture (saves ~5GB) |
| `--prompt` | string | (see below) | Prompt for detailed description |
| `--summary-prompt` | string | (see below) | Prompt for one-line summary |
| `--upload-image` | string | `Y` | Upload source image each cycle |

### Default Prompts

**Description prompt** (--prompt):
"Describe this outdoor scene in detail. Include: weather/sky conditions,
visible objects and their approximate counts, any people or animals,
notable features, and time of day estimate. Be factual and concise —
2-3 sentences."

**Summary prompt** (--summary-prompt):
"Provide a one-line summary (under 15 words) of what is happening in
this image."

### Key Parameter Interactions

- **`--gpu-mem-frac` and hardware**: This is the most critical tuning
  parameter.  On unified memory systems (Thor, DGX Spark), the GPU shares
  128 GB with the OS and other processes.  Setting 0.58 requests ~70 GB,
  leaving room for the OS.  On discrete GPUs (A100 80GB), use 0.85.

- **`--enforce-eager` and memory**: CUDA graph capture pre-allocates ~5 GB
  for execution plans.  With a 32B model already using 67 GB, this can
  cause OOM on 128 GB unified systems.  `--enforce-eager` disables graph
  capture, trading ~10% inference speed for ~5 GB memory savings.

- **`--max-model-len` and memory**: Context length directly affects KV
  cache size.  4096 tokens is sufficient for single-image description.
  Increasing to 8192 or 16384 requires proportionally more GPU memory.

- **`--temperature` and consistency**: For scientific observation, lower
  temperature (0.1–0.3) produces more consistent, factual descriptions.
  Higher values (0.7–1.0) produce more varied but potentially hallucinated
  language.

---

## 5. Understanding the Sidecar Architecture

Unlike YOLO and BioCLIP which load models in-process, this plugin uses a
**sidecar pattern**: two cooperating processes inside the same container.

```
┌─────────────────────────────────────────┐
│ Container                               │
│                                         │
│  ┌──────────────┐    ┌───────────────┐  │
│  │ app.py       │    │ vLLM Server   │  │
│  │ (plugin)     │───→│ (sidecar)     │  │
│  │              │HTTP│               │  │
│  │ • Camera     │    │ • Model loaded│  │
│  │ • Publishing │    │ • PagedAttn   │  │
│  │ • Scheduling │    │ • KV cache    │  │
│  └──────────────┘    └───────────────┘  │
│       ↓                    ↓            │
│    pywaggle             GPU memory      │
│    data store           (~70 GB)        │
└─────────────────────────────────────────┘
```

### Why a sidecar instead of direct model loading?

1. **vLLM's PagedAttention** manages GPU memory far more efficiently than
   naive `model.generate()` calls.  It pages KV cache blocks like an OS
   pages virtual memory.

2. **Server lifecycle is independent** — if the plugin crashes, the model
   stays loaded (no 7-minute reload).  If inference hangs, the plugin can
   timeout and retry.

3. **Standard API** — the OpenAI-compatible endpoint means you can test
   with curl, swap models, or even point `--vllm-url` at a remote server.

4. **Concurrency** — vLLM handles batching internally.  Multiple plugins
   could share one vLLM server (via `--vllm-url`).

### The `--vllm-url` Escape Hatch

If you set `--vllm-url http://some-host:8000`, the plugin skips
`launch_vllm_server()` entirely and connects to an existing server.
Use cases:
- Testing locally with a pre-started vLLM instance
- Sharing one vLLM server across multiple plugins on the same node
- Pointing at a remote GPU server from a CPU-only node

---

## 6. Local Testing (No Node Required)

### Prerequisites

You need a machine with:
- NVIDIA GPU with 128 GB unified memory (DGX Spark, Thor) OR 80 GB+ discrete
- CUDA 12.x+ installed
- Python 3.10+

### Quick Start

```bash
# 1. Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Redirect output to local files
export PYWAGGLE_LOG_DIR=./test-output

# 3. Start vLLM server manually (recommended for testing)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-32B-Instruct \
    --port 8199 \
    --gpu-memory-utilization 0.58 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --enforce-eager \
    --no-enable-log-requests

# 4. Wait for "Application startup complete" (~7 minutes first time)

# 5. In another terminal, run single-shot:
export PYWAGGLE_LOG_DIR=./test-output
python3 app.py \
    --stream test-image.jpg \
    --vllm-url http://localhost:8199 \
    --model Qwen/Qwen3-VL-32B-Instruct \
    --continuous N

# 6. Inspect results
cat test-output/data.ndjson
```

### What the Output Looks Like

```json
{"timestamp":"...","name":"env.scene.description","value":"Partly cloudy sky over a residential street. Two cars parked along the curb, a cyclist heading east. Mid-morning light based on shadow angles.","meta":{"camera":"test-image.jpg","model":"Qwen/Qwen3-VL-32B-Instruct"}}
{"timestamp":"...","name":"env.scene.summary","value":"Quiet residential street with parked cars and a cyclist.","meta":{"camera":"test-image.jpg","model":"Qwen/Qwen3-VL-32B-Instruct"}}
```

### Running the Test Suite

```bash
cd /path/to/Sage-agents/plugins/vllm-edge-inference
source ../../tests/.venv/bin/activate

# Unit test (no GPU needed, mocks vLLM HTTP calls)
python3 -m pytest tests/test_vllm.py -v

# Integration test (GPU required, starts real vLLM server)
# WARNING: takes ~10 minutes for model loading + warmup
python3 -m pytest tests/test_vllm_integration.py -v
```

### Testing with curl (Useful for Debugging)

Once the vLLM server is running, you can test directly:

```bash
# Health check
curl http://localhost:8199/health

# List loaded models
curl http://localhost:8199/v1/models

# Send a test image (base64-encoded)
BASE64=$(base64 -w0 test-image.jpg)
curl -X POST http://localhost:8199/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen3-VL-32B-Instruct\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$BASE64\"}},
        {\"type\": \"text\", \"text\": \"What do you see?\"}
      ]
    }],
    \"max_tokens\": 256
  }"
```

---

## 7. Deployment on a Sage Node

### Docker Build

The Dockerfile is based on `nvcr.io/nvidia/pytorch:24.06-py3` (NVIDIA's
PyTorch container with CUDA pre-installed):

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.06-py3
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Pre-download Qwen3-VL-32B-Instruct (~67 GB)
RUN mkdir -p /hf_cache && \
    HF_HOME=/hf_cache huggingface-cli download \
    Qwen/Qwen3-VL-32B-Instruct \
    --cache-dir /hf_cache --resume

ENV HF_HOME=/hf_cache
ENTRYPOINT ["python", "/app/app.py"]
```

**Warning**: The resulting image is ~80+ GB because the 67 GB model is
baked in.  This is intentional — edge nodes may lack internet access.
Build and push from a machine with fast upload:

```bash
docker build -t registry.sagecontinuum.org/waggle/vllm-edge-inference:0.1.0 .
docker push registry.sagecontinuum.org/waggle/vllm-edge-inference:0.1.0
```

### Job YAML Example

A ready-to-use job file is included at `jobs/vllm-scene-job.yaml`.
Here is the structure:

```yaml
name: vllm-scene-description
plugins:
  - name: vllm-edge-inference
    pluginSpec:
      image: registry.sagecontinuum.org/waggle/vllm-edge-inference:0.1.0
      args:
        - "--stream"
        - "bottom_camera"
        - "--interval"
        - "120"
        - "--gpu-mem-frac"
        - "0.58"
        - "--enforce-eager"
        - "--max-model-len"
        - "4096"
      selector:
        resource.gpu: "true"
        resource.gpu.memory: "128GB"
nodes:
  W097:
successcriteria:
  - WallClock('7day')
```

**Note**: This plugin requires exclusive GPU access — do not co-deploy
with other GPU-heavy plugins (BioCLIP is fine at ~1.5 GB, but running
YOLO11x simultaneously may cause memory pressure).

---

## 8. Data Published

| Topic | Type | Example | Description |
|-------|------|---------|-------------|
| `env.scene.description` | string | "Partly cloudy sky over a residential..." | Multi-sentence scene description |
| `env.scene.summary` | string | "Quiet street with parked cars" | One-line summary (under 15 words) |

Plus one uploaded JPEG per cycle: the source image for verification, with
the description stored in the upload's metadata.

### Metadata on Each Measurement

```json
{"camera": "bottom_camera", "model": "Qwen/Qwen3-VL-32B-Instruct"}
```

### Why Two Measurements?

The **description** (2-3 sentences) provides rich detail for human review
and downstream NLP analysis.  The **summary** (under 15 words) is designed
for dashboards, alerts, and quick scanning of long time series.

---

## 9. Querying Your Data

```python
import sage_data_client

# Get all scene descriptions from last 6 hours
df = sage_data_client.query(
    start="-6h",
    filter={"name": "env.scene.description", "vsn": "W097"}
)
for _, row in df.iterrows():
    print(f"{row['timestamp']}: {row['value'][:100]}...")

# Get summaries for a quick overview
df_sum = sage_data_client.query(
    start="-24h",
    filter={"name": "env.scene.summary", "vsn": "W097"}
)
print(df_sum[["timestamp", "value"]])
```

### Downstream NLP Analysis Ideas

Because the output is natural language, you can do things impossible
with structured CV outputs:

- **Keyword search**: `df[df['value'].str.contains('rain|umbrella|wet')]`
- **Sentiment/activity level**: count action words per description
- **Change detection**: compare consecutive descriptions for scene changes
- **Multi-modal fusion**: combine with YOLO counts and BioCLIP species
  for richer environmental narratives

---

## 10. Key Issues and Pitfalls

### Model Loading is Slow

Qwen3-VL-32B-Instruct takes ~7 minutes to load from cached weights on
DGX Spark/Thor.  First download from HuggingFace takes much longer (67 GB).
The Dockerfile bakes the model into the image to avoid download at runtime.
The plugin's `wait_for_ready(max_wait=600)` handles this, but be patient.

### GPU Memory Fraction is Hardware-Specific

The `--gpu-mem-frac` default (0.58) is tuned for 128 GB unified memory
systems where the OS consumes ~55 GB.  On discrete GPUs (A100 80GB), use
0.85.  Setting it too high causes OOM; too low wastes capacity.

Symptoms of wrong value:
- **Too high**: vLLM crashes during model loading with CUDA OOM
- **Too low**: vLLM warns "insufficient KV cache memory" and limits
  concurrency or refuses to start

### CUDA Graph Capture OOMs

CUDA graphs pre-record GPU operations for faster replay but consume ~5 GB
extra memory.  With a 32B model on 128 GB unified memory, this pushes past
available GPU memory.  Always use `--enforce-eager` on Thor/DGX Spark.

Symptom: vLLM starts, loads model, then crashes during warmup with
"CUDA out of memory" at the graph capture stage.

### The --disable-log-requests Flag Was Removed

vLLM v0.23.0 renamed this flag to `--no-enable-log-requests`.  Using the
old name causes an immediate crash with an unrecognized argument error.
The app.py already uses the correct flag.

### Triton JIT Compilation on First Inference

The first inference after model load triggers Triton kernel compilation
(~113 seconds on DGX Spark).  Subsequent inferences are fast (~26 seconds
for a detailed description).  This is a one-time cost per server restart.

### stdout PIPE Deadlock

If the vLLM subprocess writes to a stdout PIPE that nobody reads, the
pipe buffer fills (64 KB) and the process hangs.  The current code routes
stdout to DEVNULL and stderr to PIPE.  If you modify the launcher, be
aware of this — always drain pipes or redirect to files/DEVNULL.

### Image Encoding Overhead

Each image is JPEG-encoded, base64-encoded, and sent as part of the JSON
payload.  A 1920x1080 frame becomes ~200 KB JPEG → ~270 KB base64.  This
is well within HTTP limits but adds latency for large images.

### meta Values Must Be Strings

Same as all pywaggle plugins: all `meta={}` values must be strings.  The
uploaded image's metadata includes `description[:200]` (truncated) because
full descriptions could be very long.

### Connection Recovery

The main loop catches `requests.exceptions.ConnectionError` and retries
after 30 seconds.  This handles transient vLLM server restarts.  Other
exceptions are logged but don't kill the loop.

---

## 11. Example Scenarios

### Scenario A: Environmental Monitoring Station

```bash
python3 app.py \
    --stream bottom_camera \
    --interval 120 \
    --enforce-eager \
    --gpu-mem-frac 0.58 \
    --prompt "Describe the current weather and environmental conditions visible in this image. Include sky conditions, precipitation, visibility, wind indicators (flags, trees), and surface conditions (wet, dry, snow). Be factual — 2-3 sentences."
```

Produces a weather-log-style narrative every 2 minutes.  Useful for
correlating with BME680 sensor readings.

### Scenario B: Traffic and Activity Monitoring

```bash
python3 app.py \
    --stream bottom_camera \
    --interval 60 \
    --enforce-eager \
    --prompt "Count and describe all vehicles, pedestrians, and cyclists visible. Note their direction of travel and any notable behaviors (jaywalking, stopping, u-turns). Include approximate counts. 2-3 sentences." \
    --summary-prompt "Summarize traffic activity in under 10 words."
```

### Scenario C: Using a Smaller Model

For nodes with less GPU memory, swap to an 8B model:

```bash
python3 app.py \
    --stream bottom_camera \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --gpu-mem-frac 0.85 \
    --max-model-len 4096 \
    --interval 60
```

The 8B model uses ~16 GB and fits easily on 40 GB+ GPUs.  Quality is lower
but inference is ~3x faster.

### Scenario D: Custom Scientific Observation

```bash
python3 app.py \
    --stream top_camera \
    --interval 300 \
    --enforce-eager \
    --prompt "You are observing a prairie restoration site. Describe: dominant vegetation types visible, estimated ground cover percentage, any wildlife, signs of disturbance (erosion, tire tracks, grazing), and phenological stage (dormant, green-up, flowering, senescent). 3-4 sentences." \
    --summary-prompt "Phenological stage and dominant cover in under 10 words."
```

This demonstrates the key advantage of VLM over structured classifiers:
you can ask arbitrary domain-specific questions without retraining.

---

## Appendix: Performance Benchmarks (DGX Spark / Sage Thor)

Measured on NVIDIA GB10 (Blackwell), 128 GB unified memory, aarch64:

| Metric | Value |
|--------|-------|
| Model size | 62.49 GiB (BF16) |
| Server load time | 405 s (from HF cache) |
| Triton JIT warmup | 112.9 s (first inference only) |
| Detailed description | ~26.3 s avg (~97 tokens) |
| One-line summary | ~5.5 s avg (~19 tokens) |
| Total per cycle | ~32 s (description + summary) |
| GPU memory reserved | ~70.6 GiB (gpu_mem_frac=0.58) |
| KV cache capacity | ~4096 tokens (single sequence) |

### Model Alternatives

| Model | Size | GPU Memory | Speed | Quality |
|-------|------|-----------|-------|---------|
| Qwen3-VL-32B-Instruct | 67 GB | ~70 GB | ~26s | Excellent |
| Qwen3-VL-8B-Instruct | ~16 GB | ~20 GB | ~8s | Good |
| Phi-3.5-vision-instruct | ~8 GB | ~10 GB | ~5s | Moderate |

