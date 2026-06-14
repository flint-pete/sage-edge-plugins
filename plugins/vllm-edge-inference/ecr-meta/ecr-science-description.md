# vLLM Edge Inference — Vision-Language Scene Understanding

## Science

Edge computing nodes equipped with cameras collect vast volumes of visual
data, but most of it is never analysed in real-time.  Traditional approaches
either stream raw video to the cloud — consuming bandwidth and introducing
latency — or rely on narrow task-specific models that miss the broader
context of a scene.  Vision-language models (VLMs) bridge this gap by
generating rich, natural-language descriptions of what a camera sees,
enabling semantic understanding without custom training for each deployment.

By running a 32-billion-parameter VLM directly on 128 GB GPU-equipped edge
nodes, this plugin eliminates the cloud round-trip entirely.  The result is
real-time scene narration: every few minutes, the node captures a frame,
describes what it sees in natural language, and publishes both a detailed
description and a one-line summary to the Sage data store.

## Model

The default model is **Qwen3-VL-32B-Instruct** (~67 GB in BF16), a
state-of-the-art vision-language model from Alibaba's Qwen team.  It
accepts images and text prompts, producing fluent multi-sentence
descriptions.  The model runs under **vLLM** — a high-throughput inference
engine with PagedAttention, continuous batching, and an OpenAI-compatible
API.

On 128 GB unified memory nodes (Sage Thor / DGX Spark with NVIDIA Blackwell
GB10), the 32B model loads in ~7 minutes and generates ~97 tokens per image
description in ~26 seconds.

## Architecture: Sidecar Pattern

The plugin uses a **sidecar architecture**: it launches vLLM as a subprocess
serving an OpenAI-compatible HTTP API on localhost, then communicates with it
via standard REST calls.  This decouples the model-serving infrastructure
from the Waggle publish/upload logic, making it straightforward to swap
models or point to an external vLLM server.

```
┌─────────────────────────────────┐
│  Sage Node (128 GB unified GPU) │
│                                 │
│  ┌──────────┐   HTTP    ┌─────┐ │
│  │ app.py   │ ───────── │vLLM │ │
│  │ (plugin) │ localhost │(sub)│ │
│  └──────────┘           └─────┘ │
│       │                    │    │
│   publish()            GPU mem  │
│   upload()             (0.58)   │
└─────────────────────────────────┘
```

## Measurements Published

| Topic                  | Type   | Description                              |
|------------------------|--------|------------------------------------------|
| `env.scene.description`| string | Multi-sentence scene description         |
| `env.scene.summary`    | string | One-line summary (< 15 words)            |

Source JPEG images are uploaded each cycle for cross-referencing.

## Key Configuration for Unified Memory

| Parameter          | Unified (128 GB) | Discrete (80 GB) |
|--------------------|-------------------|-------------------|
| `--gpu-mem-frac`   | 0.58              | 0.85              |
| `--enforce-eager`  | required          | optional          |
| `--dtype`          | bfloat16          | bfloat16 / half   |

## Example Use Cases

- **Environmental scene logging** — generate a textual record of changing
  conditions (weather, traffic, construction) every 2 minutes.
- **Anomaly narration** — pair with YOLO to trigger VLM descriptions only
  when unusual objects are detected.
- **Accessibility** — provide natural-language descriptions of outdoor
  scenes for visually impaired users via connected applications.
- **Scientific field notes** — automatic captioning of ecological camera
  traps, reducing manual review burden.
