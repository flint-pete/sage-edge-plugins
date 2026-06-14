# YOLO Object Counter — Plugin Overview

A comprehensive tutorial for understanding, configuring, and deploying the
YOLO Object Counter plugin on the Sage Continuum edge computing platform.

---

## Table of Contents

1. [What This Plugin Does](#what-this-plugin-does)
2. [File Layout](#file-layout)
3. [How It Works (Internals)](#how-it-works-internals)
4. [Configuration Reference](#configuration-reference)
5. [Local Testing (No Node Required)](#local-testing)
6. [Deployment on a Sage Node](#deployment-on-a-sage-node)
7. [Data Published](#data-published)
8. [Querying Your Data](#querying-your-data)
9. [Choosing the Right Model](#choosing-the-right-model)
10. [Key Issues and Pitfalls](#key-issues-and-pitfalls)
11. [Example Scenarios](#example-scenarios)

---

## 1. What This Plugin Does

The YOLO Object Counter captures frames from a camera attached to a Sage
edge node, runs the YOLO11x object detection model on the GPU, counts how
many of each object class appear in the frame, and publishes those counts
to the Sage data store.  Optionally, it uploads an annotated image with
bounding boxes drawn around each detected object.

In concrete terms: every N seconds the node says "I see 3 people, 2 cars,
and 1 bicycle" — and that becomes a queryable time-series record in the Sage
data API.

**Why at the edge?**  Streaming raw video to the cloud costs bandwidth
(~5 Mbps per 1080p stream), introduces latency, and raises privacy concerns.
YOLO inference takes <50 ms on the GPU, so we can publish compact counts
instead of raw frames.

---

## 2. File Layout

```
yolo-object-counter/
├── app.py                          # Main application (175 lines)
├── Dockerfile                      # Container build instructions
├── requirements.txt                # Python dependencies
├── sage.yaml                       # ECR metadata — name, version, inputs
├── overview.md                     # This file
└── ecr-meta/                       # ECR submission metadata
    ├── README                      # Instructions for ECR submission
    ├── ecr-science-description.md  # Science narrative (displayed on ECR portal)
    ├── ecr-credits-license.txt     # Authors, funding, license
    ├── ecr-project-keywords.txt    # Search keywords / ontology terms
    └── ecr-project-url.txt         # URL to project science page
    (ecr-icon.jpg)                  # 512x512 icon — create before submission
    (ecr-science-image.jpg)         # 1920x1080 science image — create before submission
```

**What each file does:**

| File | Purpose |
|------|---------|
| `app.py` | The entire plugin — model loading, camera capture, inference, publishing |
| `Dockerfile` | Builds the container image from NVIDIA PyTorch base, installs deps |
| `requirements.txt` | pip dependencies: pywaggle, ultralytics, torch, opencv, numpy, pillow |
| `sage.yaml` | ECR registry metadata: name, version, inputs, ontology, architecture |
| `ecr-meta/` | Supplementary metadata displayed on the ECR portal page |

---

## 3. How It Works (Internals)

### Startup Sequence

```
1. Parse command-line arguments (argparse)
2. Create YOLODetector instance
   └── Load YOLO model weights (auto-downloads if not cached)
   └── Move model to GPU (CUDA) or CPU
3. Open camera stream via pywaggle Camera abstraction
4. Enter main loop
```

### Main Loop (one iteration)

```
┌─────────────────────────────────────────────────────┐
│ 1. cam.snapshot()                                   │
│    └── Captures a single BGR numpy frame            │
│                                                     │
│ 2. detector.detect(frame, target_classes)            │
│    └── Runs YOLO inference on GPU                   │
│    └── Filters by confidence threshold              │
│    └── Applies NMS (non-maximum suppression)        │
│    └── Optionally filters to target classes only    │
│    └── Returns list of {class, confidence, bbox}    │
│                                                     │
│ 3. Count detections per class                       │
│    └── collections.Counter on class names            │
│                                                     │
│ 4. plugin.publish() for each class                  │
│    └── "env.count.person" → 3                       │
│    └── "env.count.car" → 2                          │
│    └── "env.count.total" → 5                        │
│    └── meta: model, confidence threshold, camera     │
│                                                     │
│ 5. If --upload-image Y:                              │
│    └── draw_boxes() — annotates frame with boxes     │
│    └── Save to temp JPEG                             │
│    └── plugin.upload_file() — sends to object store  │
│                                                     │
│ 6. Sleep(interval) and repeat (if continuous=Y)      │
└─────────────────────────────────────────────────────┘
```

### Key Classes

**`YOLODetector`** (lines 36–73)
- Wraps the Ultralytics YOLO API
- `__init__`: loads model, moves to CUDA, stores thresholds
- `detect(frame, target_classes)`: runs inference, returns list of dicts
- Each detection dict: `{"class": "person", "confidence": 0.87, "bbox": [x1, y1, x2, y2]}`

**`draw_boxes()`** (lines 76–88)
- Takes the raw frame + detections list
- Draws green rectangles and labels using OpenCV
- Returns a new annotated frame (does not modify the original)

### pywaggle Integration

The plugin uses two pywaggle abstractions:

1. **`Camera(stream)`** — abstracts camera access.  The `stream` argument
   can be:
   - A named camera: `"bottom_camera"`, `"top_camera"` (resolved by node config)
   - An RTSP URL: `"rtsp://192.168.1.100:554/stream"`
   - A file path: `"/path/to/test-image.jpg"` (for local testing)

2. **`Plugin()`** — handles data publishing.  Used as a context manager:
   - `plugin.publish(topic, value, meta={...})` — publishes a measurement
   - `plugin.upload_file(path)` — uploads a file to the object store
   - When `PYWAGGLE_LOG_DIR` is set, writes to local files instead of Beehive

**Critical rule**: All `meta={}` values must be strings.  `{"count": str(n)}`
not `{"count": n}`.  pywaggle's `valid_meta()` enforces this and will raise
ValueError on non-string values.

---

## 4. Configuration Reference

All parameters are passed as command-line arguments:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--stream` | string | `bottom_camera` | Camera source |
| `--model` | string | `yolo11x.pt` | YOLO model name |
| `--interval` | int | `30` | Seconds between captures |
| `--conf-thres` | float | `0.25` | Minimum detection confidence |
| `--iou-thres` | float | `0.45` | NMS IoU threshold |
| `--classes` | string | `""` (all) | Comma-separated target classes |
| `--continuous` | string | `Y` | `Y` = loop, `N` = single-shot |
| `--upload-image` | string | `Y` | Upload annotated images |

### The `--classes` Filter

YOLO11x recognises 80 COCO classes.  By default all are counted.  To count
only specific classes, pass a comma-separated list:

```bash
--classes "person,car,truck,bus,bicycle"     # traffic monitoring
--classes "bird"                              # avian counting
--classes "dog,cat"                           # pet detection
```

Class names must match COCO names exactly (lowercase).  Common classes:
`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `bird`, `cat`,
`dog`, `horse`, `sheep`, `cow`, `backpack`, `umbrella`, `handbag`,
`suitcase`, `bottle`, `chair`, `bench`, `potted plant`.

### Confidence Threshold

The `--conf-thres` parameter controls the minimum confidence for a detection
to be included.  Lower values detect more objects but increase false
positives:

| Threshold | Use Case |
|-----------|----------|
| 0.10 | Maximum recall — detect everything (noisy) |
| 0.25 | Default — good balance |
| 0.50 | High precision — only confident detections |
| 0.70 | Very conservative — few false positives |

---

## 5. Local Testing (No Node Required)

pywaggle supports local testing by redirecting all publish/upload calls to
local files.  No Sage node, no Docker, no credentials needed.

### Quick Start

```bash
# 1. Set up a Python virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Tell pywaggle to write output locally
export PYWAGGLE_LOG_DIR=./test-output

# 3. Run single-shot against a test image
python3 app.py --stream test-image.jpg --continuous N

# 4. Check results
cat test-output/data.ndjson          # Published measurements (one JSON per line)
ls test-output/uploads/              # Annotated images
```

### What `PYWAGGLE_LOG_DIR` Does

When this environment variable is set, pywaggle intercepts all
`plugin.publish()` and `plugin.upload_file()` calls and writes them locally:

- **`data.ndjson`** — Newline-delimited JSON, one record per publish call:
  ```json
  {"timestamp":"2025-06-12T10:30:00Z","name":"env.count.person","value":3,"meta":{"model":"yolo11x.pt","camera":"test-image.jpg"}}
  ```
- **`uploads/`** — Uploaded files saved as `{timestamp}-{filename}`

### Running the Test Suite

```bash
cd /path/to/Sage-agents
source tests/.venv/bin/activate

# Unit test (mocked model, no GPU)
python3 -m pytest tests/test_yolo.py -v

# Integration test (real model, GPU required)
python3 -m pytest tests/test_yolo_integration.py -v
```

---

## 6. Deployment on a Sage Node

### Building the Docker Image

```bash
# On a machine with Docker:
docker build -t registry.sagecontinuum.org/waggle/yolo-object-counter:0.1.0 .
docker push registry.sagecontinuum.org/waggle/yolo-object-counter:0.1.0
```

### Submitting a Job

Create a job YAML (see `../jobs/yolo-counter-job.yaml`):

```yaml
name: yolo-bird-counter
plugins:
  - name: yolo-object-counter
    pluginSpec:
      image: registry.sagecontinuum.org/waggle/yolo-object-counter:0.1.0
      args:
        - "--stream"
        - "bottom_camera"
        - "--classes"
        - "bird"
        - "--interval"
        - "60"
        - "--conf-thres"
        - "0.30"
nodes:
  W097:
scienceRules:
  - "schedule(yolo-object-counter): cronjob('count-birds', '*/10 * * * *')"
successcriteria:
  - WallClock('7day')
```

Submit:

```bash
export SES_HOST=https://es.sagecontinuum.org
export SES_USER_TOKEN=<your-token>
sesctl create --from-file yolo-bird-counter-job.yaml
sesctl sub yolo-bird-counter
```

### On-Node Testing with pluginctl

```bash
ssh waggle-dev-node-V032
pluginctl build .
pluginctl run --name test-yolo \
    registry.sagecontinuum.org/waggle/yolo-object-counter:0.1.0 \
    -- --stream bottom_camera --classes bird --continuous N
pluginctl logs test-yolo
```

---

## 7. Data Published

Every inference cycle publishes these measurements:

```
env.count.person     → 3      meta: {model: "yolo11x.pt", camera: "bottom_camera", conf_thres: "0.25"}
env.count.car        → 2      meta: {model: "yolo11x.pt", camera: "bottom_camera", conf_thres: "0.25"}
env.count.total      → 5      meta: {model: "yolo11x.pt", camera: "bottom_camera", conf_thres: "0.25"}
```

Plus one uploaded JPEG (if `--upload-image Y`): the original frame with
bounding boxes and confidence labels drawn on it.

---

## 8. Querying Your Data

After deployment, retrieve counts using the Sage data API:

```python
import sage_data_client

df = sage_data_client.query(
    start="-1h",
    filter={"name": "env.count.*", "vsn": "W097", "plugin": "*yolo*"}
)
print(df[["timestamp", "name", "value"]].to_string())
```

Or via curl:

```bash
curl -s -X POST https://data.sagecontinuum.org/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"start":"-1h","filter":{"name":"env.count.bird","vsn":"W097"}}' \
  | python3 -m json.tool
```

---

## 9. Choosing the Right Model

| Model | Params | mAP (COCO) | Inference (GB10) | Use Case |
|-------|--------|------------|------------------|----------|
| yolo11n.pt | 2.6M | 39.5% | ~5 ms | Maximum speed, low power |
| yolo11s.pt | 9.4M | 47.0% | ~8 ms | Fast + decent accuracy |
| yolo11m.pt | 20.1M | 51.5% | ~14 ms | Balanced |
| yolo11x.pt | 56.9M | 54.7% | ~21 ms | **Default — best accuracy** |
| yolov8x.pt | 68.2M | 53.9% | ~25 ms | Alternative large model |

The model auto-downloads on first use.  To pre-download, add a `RUN` in the
Dockerfile:

```dockerfile
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

---

## 10. Key Issues and Pitfalls

### Meta Values Must Be Strings

```python
# WRONG — will raise ValueError
plugin.publish("env.count.person", 3, meta={"confidence": 0.25})

# RIGHT
plugin.publish("env.count.person", 3, meta={"confidence": "0.25"})
```

pywaggle's `valid_meta()` enforces `isinstance(v, str)` on every meta value.

### Model Download on First Run

YOLO models are downloaded from Ultralytics on first use (~110 MB for
yolo11x.pt).  On edge nodes without internet, the model must be baked into
the Docker image at build time (see the Dockerfile `RUN` download line).

### Camera Stream Names

Camera names like `bottom_camera` and `top_camera` are resolved by the
node's local configuration.  Not every node has every camera.  Check node
capabilities via `pluginctl` or the Sage portal before deploying.

### GPU vs CPU

The plugin auto-detects CUDA availability.  On CPU-only nodes, inference
is ~10x slower (200–500 ms vs 20 ms).  For CPU nodes, use a smaller model
like `yolo11n.pt`.

### OpenCV and Headless Mode

The Dockerfile installs `opencv-python` (not `opencv-python-headless`).
If you see `libGL` errors, switch to `opencv-python-headless` in
`requirements.txt`.

### NMS and Overlapping Detections

The `--iou-thres` parameter controls non-maximum suppression.  Lower values
are more aggressive at removing overlapping boxes.  If you see duplicate
detections of the same object, lower this from 0.45 to 0.30.

---

## 11. Example Scenarios

### Scenario A: Bird Counting at a Wildlife Station

```bash
python3 app.py \
    --stream bottom_camera \
    --model yolo11x.pt \
    --classes bird \
    --interval 60 \
    --conf-thres 0.30 \
    --upload-image Y
```

This captures a frame every 60 seconds, counts only birds, and uploads
annotated images.  The higher confidence threshold (0.30) reduces false
positives from distant or partially occluded birds.

### Scenario B: Urban Traffic Monitoring

```bash
python3 app.py \
    --stream rtsp://192.168.1.100:554/traffic \
    --classes "person,car,truck,bus,bicycle,motorcycle" \
    --interval 30 \
    --conf-thres 0.25
```

### Scenario C: Single-Shot Testing

```bash
export PYWAGGLE_LOG_DIR=./test-output
python3 app.py \
    --stream /path/to/parking-lot.jpg \
    --classes car \
    --continuous N
cat test-output/data.ndjson
```

---

## Appendix: Performance Benchmarks (DGX Spark / Sage Thor)

Measured on NVIDIA GB10 (Blackwell), 128 GB unified memory, aarch64:

| Metric | Value |
|--------|-------|
| Model load time | 2.71 s |
| Warmup inference | 0.74 s |
| Average inference | 20.6 ms |
| GPU memory | ~1.2 GB |

These benchmarks show that YOLO11x is extremely lightweight relative to the
128 GB available on Thor nodes, leaving ample room for concurrent plugins
(BioCLIP, vLLM).
