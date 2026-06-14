# Sage Edge Plugin Runtime & Packaging Tutorial

> **Audience**: Developers building ML plugins for the Sage Continuum platform.
> **Goal**: Understand the one-shot scheduling model, container caching, and how
> to package plugins so cold-start time is measured in seconds, not minutes.

---

## Table of Contents

1. [The One-Shot Execution Model](#1-the-one-shot-execution-model)
2. [Container Lifecycle on the Node](#2-container-lifecycle-on-the-node)
3. [Image Caching and Layer Reuse](#3-image-caching-and-layer-reuse)
4. [Model Packaging Strategies](#4-model-packaging-strategies)
5. [Cold-Start Optimization Playbook](#5-cold-start-optimization-playbook)
6. [Lessons from Production Plugins](#6-lessons-from-production-plugins)
7. [Plan for Our Three Plugins](#7-plan-for-our-three-plugins)

---

## 1. The One-Shot Execution Model

Sage plugins are **not** long-running daemons. The fundamental scheduling
paradigm is **one-shot**: the scheduler fires a container, it does its work,
and it exits. This is enforced by design for three reasons:

| Reason             | Why it matters                                       |
|--------------------|------------------------------------------------------|
| **Fair sharing**   | Multiple science goals share the same GPU/CPU/RAM    |
| **Resilience**     | A crashed plugin is automatically rescheduled        |
| **Reproducibility**| Each run is stateless and independent                |

### How it works in practice

```
┌──────────────────────────────────────────────────────────┐
│ Edge Scheduler (cloud)                                   │
│  - Receives "Science Goal" YAML from user                │
│  - Distributes job specs to target nodes                 │
│  - Monitors status, handles failures                     │
└────────────────┬─────────────────────────────────────────┘
                 │ polls
                 ▼
┌──────────────────────────────────────────────────────────┐
│ WES / k3s (on-node)                                      │
│  1. Node agent receives job                              │
│  2. containerd pulls image (if not cached)               │
│  3. k3s creates a Kubernetes CronJob / Job               │
│  4. Pod runs: init → process → publish → exit            │
│  5. Pod marked Completed                                 │
│  6. Completed pods cleaned up                            │
└──────────────────────────────────────────────────────────┘
```

### Three scheduling modes

| Mode        | YAML                        | Behavior                              |
|-------------|-----------------------------|---------------------------------------|
| **Cronjob** | `schedule: "*/10 * * * *"`  | New one-shot pod every 10 min         |
| **Lambda**  | `when: {name: ..., cond: ...}` | Pod fires in response to data event |
| **Always**  | `schedule: "always"`        | Continuous (rare, discouraged)        |

The cronjob mode is by far the most common. Each cron tick creates a fresh
Kubernetes Job. The container starts, does its work, publishes results via
pywaggle, and exits. The restart policy is `Never` — if it crashes, it stays
crashed for that tick and the next cron tick creates a new pod.

### What this means for your plugin

Your `app.py` must follow the **Start → Process → Publish → Exit** pattern:

```python
from waggle.plugin import Plugin
from waggle.data.vision import Camera

with Plugin() as plugin:
    with Camera(args.stream) as cam:
        sample = cam.snapshot()
        # ... run inference ...
        plugin.publish("env.count.car", count, timestamp=sample.timestamp)
        # exit naturally — the container is done
```

If you need multiple frames, loop with a fixed iteration count or timeout,
then exit. The scheduler will fire you again at the next cron tick.

---

## 2. Container Lifecycle on the Node

Each Sage Wild node runs **k3s** (lightweight Kubernetes) with **containerd**
as the container runtime. Understanding the lifecycle helps you optimize for
fast starts and minimal resource waste.

### Pod lifecycle timeline

```
Time ─────────────────────────────────────────────────────►

│ Image Pull  │ Container  │   Your app.py    │ Cleanup │
│ (if needed) │   Start    │   runs here      │         │
│             │            │                  │         │
├─────────────┼────────────┼──────────────────┼─────────┤
│  0-300s     │  1-5s      │  10s - 5min      │  ~1s    │
│  (first run)│            │  (depends on     │         │
│  0s         │            │   workload)      │         │
│  (cached)   │            │                  │         │
```

### What happens at each stage

**Image Pull** — containerd checks its local image store. If the image
(by tag + digest) is already present, this step is instant. On first run,
it downloads every layer. For a 15 GB ML image, this can take minutes on
a constrained uplink. After that first pull, it's cached.

**Container Start** — k3s creates the pod, mounts volumes (if any), sets
environment variables, and starts the entrypoint. The `waggle/plugin-base`
images set `ENTRYPOINT ["python3", "/app/app.py"]`, so your script runs
immediately.

**Your app.py** — This is where inference happens. You have access to:
- GPU via NVIDIA device plugin (if your job requests it)
- Camera streams via `waggle.data.vision.Camera`
- Publishing via `waggle.plugin.Plugin`
- Local filesystem (ephemeral — gone when pod exits)

**Cleanup** — Pod exits, status set to `Completed`. k3s garbage-collects
completed pods after a retention period. The image stays cached.

### Resource requests

Jobs can request specific resources in the science goal YAML:

```yaml
plugins:
  - name: my-plugin
    pluginSpec:
      image: registry.sagecontinuum.org/user/my-plugin:0.1.0
      resource:
        request:
          cpu: "4"
          memory: "8Gi"
        limit:
          cpu: "8"
          memory: "16Gi"
          gpu: "1"
```

The node scheduler won't start your pod until requested resources are
available. This is why the one-shot model matters: GPU-heavy plugins
finish and release the GPU for the next scheduled plugin.

---

## 3. Image Caching and Layer Reuse

This is the single most important optimization concept for Sage plugins.

### How containerd caching works

containerd stores images as **content-addressable layers**. When you pull
`registry.sagecontinuum.org/user/my-plugin:0.1.0`, containerd:

1. Fetches the image manifest (tiny — just layer digests)
2. For each layer, checks if the digest exists locally
3. Only downloads layers it doesn't already have
4. Assembles the image from cached + new layers

### Why this matters for ML plugins

A typical ML plugin image stack looks like:

```
Layer 1:  Ubuntu base              (~80 MB)   ← shared across plugins
Layer 2:  CUDA runtime             (~3 GB)    ← shared across GPU plugins
Layer 3:  PyTorch + dependencies   (~5 GB)    ← shared if same base
Layer 4:  Your code + pip packages (~500 MB)  ← plugin-specific
Layer 5:  Baked model weights      (~1-20 GB) ← plugin-specific
```

If two plugins share the same base image (e.g., both use
`nvcr.io/nvidia/pytorch:24.06-py3`), layers 1-3 are pulled once and
reused. Only layers 4-5 need downloading for the second plugin.

### The imagePullPolicy trap

k3s defaults:

| Image tag          | Default policy    | Behavior                    |
|--------------------|-------------------|-----------------------------|
| `my-plugin:0.1.0`  | `IfNotPresent`    | Pull once, cache forever    |
| `my-plugin:latest` | `Always`          | Check registry every time   |
| `my-plugin@sha256` | `IfNotPresent`    | Pull once (immutable)       |

**Always use explicit version tags, never `:latest`** for production
plugins. With `:latest`, every cron tick triggers a registry check (and
potentially a re-pull), adding seconds or minutes to every run.

### Layer ordering matters

Dockerfile layers are cached top-down. If a layer changes, every layer
below it is invalidated. The optimal ordering for ML plugins:

```dockerfile
# ✅ GOOD: Model weights early, code last
FROM nvcr.io/nvidia/pytorch:24.06-py3           # Layer 1-3 (shared)
COPY requirements.txt /app/                      # Layer 4a (changes rarely)
RUN pip install -r /app/requirements.txt         # Layer 4b (changes rarely)
ADD https://models.example.com/yolo.pt /app/     # Layer 5  (changes rarely)
COPY app.py /app/                                # Layer 6  (changes often)

# ❌ BAD: Code before model
FROM nvcr.io/nvidia/pytorch:24.06-py3
COPY app.py /app/                                # ← changes often
ADD https://models.example.com/yolo.pt /app/     # ← re-downloaded every rebuild!
```

When you change `app.py` and rebuild, only layer 6 is new in the good
example. In the bad example, changing `app.py` invalidates the model
download layer too, forcing a complete re-download.

---

## 4. Model Packaging Strategies

Every production Sage plugin we studied bakes model weights into the Docker
image at build time. No exceptions. Here's what each does:

### Pattern A: Direct URL download with ADD

Used by: **sound-event-detection**, **avian-diversity-monitoring**

```dockerfile
ADD https://web.lcrc.anl.gov/public/waggle/models/yamnet.tflite /app/yamnet.tflite
ADD https://web.lcrc.anl.gov/public/waggle/models/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite /app/
```

Pros: Simple, one line, Docker caches the layer.
Cons: URL must be stable. If it goes down, builds fail.

### Pattern B: Download via RUN command

Used by: **object-counter** (YOLOv7)

```dockerfile
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -O /app/yolov7.pt
```

Pros: Can add retry logic, checksums, error handling.
Cons: Slightly more verbose.

### Pattern C: Framework auto-download at build time

Used by: **our yolo-object-counter**

```dockerfile
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Pros: Uses the framework's built-in model registry (Ultralytics Hub,
HuggingFace, etc.). Handles checksums and caching automatically.
Cons: Need to know where the framework caches the file so you can
find it at runtime.

### Pattern D: COPY from build context

For models too large for URL download or hosted privately:

```dockerfile
COPY ./models/my-model.pt /app/models/my-model.pt
```

Pros: Works with any model source. No network dependency at build time
(after initial download to your dev machine).
Cons: Model must be in the Docker build context. The `.pt` file should
be `.gitignore`'d and downloaded separately.

### What NOT to do

```python
# ❌ NEVER download models at runtime
def main():
    model = YOLO("yolov8n.pt")  # Downloads on first run!
```

This fails because:
1. Edge nodes may have limited or no internet access
2. Every cron tick re-downloads the model (if no persistent storage)
3. Pods are ephemeral — there's no persistent filesystem between runs
4. Multiple concurrent runs fight over bandwidth

---

## 5. Cold-Start Optimization Playbook

On a 10-minute cron schedule, your plugin gets ~600 seconds per cycle.
If cold-start takes 60 seconds, you've lost 10% of your window. Here's
how to minimize it.

### 5.1 Minimize image size where possible

While ML images are inherently large (5-20 GB), you can trim fat:

```dockerfile
# Remove pip cache
RUN pip install --no-cache-dir -r requirements.txt

# Remove build artifacts
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Use multi-stage builds for compiled dependencies
FROM builder AS build-stage
RUN pip install --prefix=/install -r requirements.txt
FROM runtime AS final
COPY --from=build-stage /install /usr/local
```

### 5.2 Optimize Python imports

Heavy imports (torch, ultralytics, transformers) can take 5-15 seconds.
Profile your startup:

```python
import time
t0 = time.time()
import torch
print(f"torch import: {time.time()-t0:.1f}s")  # Often 3-5s
```

### 5.3 Use the smallest model that meets your science goal

| Model              | Size   | Load time | Accuracy | Use case          |
|--------------------|--------|-----------|----------|-------------------|
| YOLOv8n            | 6 MB   | ~1s       | Good     | Quick counting    |
| YOLOv8x            | 130 MB | ~3s       | Best     | Detailed analysis |
| BioCLIP2 (ViT-B)   | 600 MB | ~5s       | Good     | Species ID        |
| vLLM (Qwen2.5-VL)  | 5-70GB | 30-120s   | Best     | Scene description |

### 5.4 Warm up the model before processing

Some frameworks compile or optimize on first inference. Do a warmup:

```python
# Warm up with a dummy input
dummy = torch.zeros(1, 3, 640, 640).to(device)
model(dummy)  # First inference triggers JIT compilation

# Now process real data
with Camera(args.stream) as cam:
    sample = cam.snapshot()
    results = model(sample.data)
```

### 5.5 Share base images across your plugin suite

If you maintain multiple plugins, use the same base image. On the node,
layers 1-3 are shared:

```
yolo-object-counter    ─┐
bioclip-species-id      ├── nvcr.io/nvidia/pytorch:24.06-py3 (cached once)
vllm-edge-inference    ─┘
```

Only the pip packages, code, and model layers differ per plugin.

---

## 6. Lessons from Production Plugins

We studied four production Sage plugins to extract packaging patterns.
Here's what we learned from each.

### 6.1 cloud-motion-v1 (optical flow)

**Base image**: `waggle/plugin-base:1.1.1-base` (no GPU needed)
**Model**: None — uses OpenCV optical flow (algorithmic, no ML model)
**Key patterns**:
- Uses `Camera(args.stream)` for live frames, falls back to file input
- Publishes cloud motion vectors as `env.cloud.motion.{x,y,speed,angle}`
- Runs with `--continuous` flag for multi-sample mode
- Clean separation: `compute_optical_flow()` function is pure numpy/OpenCV

**Takeaway**: Not every plugin needs a model. Pure CV algorithms work great
as lightweight plugins with tiny images (~200 MB).

### 6.2 sound-event-detection (YAMNet)

**Base image**: `waggle/plugin-base:1.1.1-base` (CPU-only TFLite)
**Model**: YAMNet TFLite (~4 MB), downloaded via `ADD` from lcrc.anl.gov
**Key patterns**:
- Audio input via PyAudio: `stream = p.open(format=..., channels=1, rate=16000)`
- Runs inference on 0.975s audio windows (YAMNet's native window)
- Publishes top-N class names: `plugin.publish("env.sound.classification", ...)`
- Uses `sage-cli.py storage files download` as fallback (commented out)

**Takeaway**: Audio plugins follow the same pattern — capture, infer, publish.
The TFLite model is tiny enough to `ADD` directly. Note the migration from
sage-cli storage to direct URL — host models on a stable URL.

### 6.3 avian-diversity-monitoring (BirdNET)

**Base image**: `nvcr.io/nvidia/l4t-tensorflow:r32.4.4` (Jetson-specific!)
**Model**: BirdNET TFLite (~25 MB) from lcrc.anl.gov
**Key patterns**:
- Combines audio (BirdNET) with camera (image capture on detection)
- Uses GPS coordinates for species filtering: `--lat 41.7 --lon -87.6`
- Threshold-based publishing: only publishes detections above confidence
- Records audio + captures image simultaneously for verification

**Takeaway**: Multi-modal plugins (audio + camera) are common. Note the
Jetson-specific base image — architecture matters. Our Thor nodes are
aarch64 like Jetson but with much more memory and a Blackwell GPU.

### 6.4 object-counter (YOLOv7)

**Base image**: `waggle/plugin-base:1.1.1-ml-torch1.9`
**Model**: YOLOv7 (~75 MB) downloaded via `wget` in Dockerfile
**Key patterns**:
- Full YOLO inference pipeline with NMS
- Publishes per-class counts: `env.count.car`, `env.count.person`, etc.
- Uses `--conf-thres` and `--iou-thres` CLI args for tuning
- Frames from `Camera(args.stream)` converted to tensor manually

**Takeaway**: Our yolo-object-counter follows this same pattern but uses
the newer Ultralytics YOLOv8+ API which is significantly cleaner.
The publish pattern (`env.count.<class>`) is the established convention.

### Common patterns across all four

| Feature               | cloud-motion | sound-event | avian-div | object-counter |
|-----------------------|:------------:|:-----------:|:---------:|:--------------:|
| waggle Plugin ctx mgr | ✓            | ✓           | ✓         | ✓              |
| argparse CLI          | ✓            | ✓           | ✓         | ✓              |
| Model baked in image  | n/a          | ✓ (ADD)     | ✓ (ADD)   | ✓ (wget)       |
| Camera input          | ✓            | —           | ✓         | ✓              |
| Audio input           | —            | ✓           | ✓         | —              |
| Ontology-based names  | ✓            | ✓           | ✓         | ✓              |
| One-shot capable      | ✓            | ✓           | ✓         | ✓              |

---

## 7. Plan for Our Three Plugins

Based on everything above, here's the concrete plan for bringing our three
plugins up to production quality and submitting them to the ECR.

### 7.1 yolo-object-counter

**Status**: Core logic works. Unit tests pass.

**What's done**:
- app.py with Ultralytics YOLOv8, pywaggle integration, argparse CLI
- Dockerfile with model baking via `RUN python3 -c "...YOLO('yolov8n.pt')"`
- Unit tests with mocked waggle
- sage.yaml, overview.md, ecr-meta/

**What needs work**:

| Item | Action | Priority |
|------|--------|----------|
| Scale model up | Add `--model` arg to select yolov8n/s/m/l/x at runtime; bake yolov8x.pt (~130 MB) for Thor's 128GB | High |
| Dockerfile layer order | Move `COPY app.py` to last layer (after model bake) | Medium |
| Integration test | Test with real model on GPU, verify inference output format | High |
| ecr-icon.jpg | Create a representative icon for ECR listing | Medium |
| ecr-science-image.jpg | Create a science image showing detection output | Medium |
| Job YAML | Verify cronjob schedule, resource requests match Thor specs | Medium |

### 7.2 bioclip-species-classifier

**Status**: Core logic works. Unit tests pass.

**What's done**:
- app.py with BioCLIP2 (open_clip), pywaggle integration
- Dockerfile with model baking via HuggingFace download
- Unit tests with mocked waggle
- sage.yaml, overview.md, ecr-meta/

**What needs work**:

| Item | Action | Priority |
|------|--------|----------|
| Scale model up | Evaluate BioCLIP2 ViT-L vs ViT-B; 128GB allows the larger model | High |
| Top-K publishing | Publish top-5 species with confidence scores, not just top-1 | High |
| Taxonomy integration | Add `--taxonomy` arg for region-specific species lists | Medium |
| Dockerfile layer order | Optimize layer ordering for iterative development | Medium |
| Integration test | Test with real model, verify species classification accuracy | High |
| ecr-icon.jpg | Create icon | Medium |
| ecr-science-image.jpg | Create science image | Medium |

### 7.3 vllm-edge-inference

**Status**: Core logic works. Unit tests pass. Most complex plugin.

**What's done**:
- app.py with vLLM for vision-language inference
- Dockerfile with model baking (Qwen2.5-VL)
- Unit tests with mocked waggle
- sage.yaml, overview.md, ecr-meta/

**What needs work**:

| Item | Action | Priority |
|------|--------|----------|
| Model selection | Evaluate Qwen2.5-VL-7B vs 72B for Thor's 128GB memory | Critical |
| Cold-start optimization | vLLM engine startup is slow (30-120s); consider keeping engine warm across frames | High |
| Prompt engineering | Refine system prompt for ecological/infrastructure observations | High |
| Output parsing | Structured JSON output for publishable measurements | High |
| Dockerfile layer order | Model layer is huge (7-70 GB); must be early | Medium |
| Integration test | Test with real vLLM engine on GPU | High |
| Memory profiling | Verify peak memory fits in Thor's 128GB with model + KV cache | Critical |
| ecr-icon.jpg | Create icon | Medium |
| ecr-science-image.jpg | Create science image | Medium |

### 7.4 Cross-cutting work

| Item | Action |
|------|--------|
| Add `*.pt` to `.gitignore` | Prevent model weights from being committed |
| Shared base image | All three use `nvcr.io/nvidia/pytorch:24.06-py3` — document this for layer sharing |
| Combined job YAML | Update `jobs/combined-ml-pipeline-job.yaml` with correct resource requests |
| README.md | Update top-level README with build/test/deploy instructions |
| ECR submission | Push all three to `registry.sagecontinuum.org` (blocked on credentials) |

### 7.5 Execution order

```
Phase 1 (Now):
  ├── Add *.pt to .gitignore
  ├── Fix Dockerfile layer ordering (all 3 plugins)
  └── Scale up model selections for 128GB Thor nodes

Phase 2 (Integration Testing):
  ├── Build Docker images locally on DGX Spark
  ├── Run integration tests with real models + GPU
  ├── Memory-profile vLLM with target model
  └── Verify publish output format matches ontology

Phase 3 (Polish):
  ├── Create ecr-icon.jpg and ecr-science-image.jpg (all 3)
  ├── Finalize overview.md with benchmark numbers
  └── Update combined job YAML

Phase 4 (Deploy):
  ├── Push images to registry.sagecontinuum.org
  ├── Submit to ECR for review
  └── Test on a real Thor node
```

---

*Generated from analysis of 4 production Sage plugins and the Sage/Waggle
documentation. Last updated: June 2025.*
