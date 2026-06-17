# Building and Deploying Plugin Docker Images

How to build Docker images for each plugin on a machine with
internet access, push them to a registry, and run them on a
Sage Thor node via `pluginctl`.

This is the officially supported Sage development workflow.
See: https://sagecontinuum.org/docs/reference-guides/dev-quick-reference


## Prerequisites

- A build machine with internet access and Docker (e.g. DGX Spark)
- SSH access to a Thor node
- The sage-edge-plugins repo cloned on both machines


## Base Image

All three plugins use `nvcr.io/nvidia/pytorch:25.04-py3`:

| Component | Version |
|-----------|---------|
| CUDA | 12.9 |
| PyTorch | 2.7 |
| Python | 3.12 |
| Ubuntu | 24.04 |
| Min driver | R575+ |

This image supports Blackwell GPUs (sm_120/sm_121) natively —
no PTX JIT compilation needed. The previous base image
(`24.06-py3`, PyTorch 2.4) only supported up to sm_90 (Hopper)
and would silently fall back to CPU on Blackwell hardware.

Both DGX Spark (GB10, driver 580.159) and Thor nodes (NVIDIA Thor,
driver 580.00) are Blackwell architecture and require this image.


## 1. Build on a Machine with Internet

Sage edge nodes do not have outbound internet access. Docker
images must be built on a machine that can reach PyPI, GitHub,
and HuggingFace, then transferred to the node.

### YOLO Object Counter (~4 GB image, ~5 min build)

```bash
cd ~/AI-projects/Sage-agents/plugins/yolo-object-counter
docker build --no-cache -t yolo-object-counter:0.1.0 .
```

### BioCLIP Species Classifier (~6 GB image, ~10 min build)

```bash
cd ~/AI-projects/Sage-agents/plugins/bioclip-species-classifier
docker build --no-cache -t bioclip-species-classifier:0.1.0 .
```

### vLLM Edge Inference (~80 GB image, ~30 min build)

```bash
cd ~/AI-projects/Sage-agents/plugins/vllm-edge-inference
docker build --no-cache -t vllm-edge-inference:0.1.0 .
```

The vLLM image is huge because it bakes in the Qwen3-VL-32B-Instruct
model (~67 GB). This is intentional — edge nodes cannot download
models at runtime.

### Dockerfile: OpenCV fix

All three Dockerfiles include this fix to ensure `opencv-python-headless`
(no GUI) is used instead of the base image's `opencv-python` (with GUI):

```dockerfile
# Remove base image opencv, install headless variant
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2* && \
    pip install --no-cache-dir opencv-python-headless>=4.8.0
```

Edge nodes are headless — the GUI variant wastes space and can cause
import conflicts. The `rm -rf cv2*` clears stale files that
`pip uninstall` sometimes leaves behind.


## 2. Test Locally on the Build Machine

Before pushing to a registry, verify the image works:

```bash
# YOLO — quick test with GPU
docker run --rm --gpus all \
    -v ~/AI-projects/Sage-agents/plugins/yolo-object-counter/tests/test-images:/images:ro \
    -e PYWAGGLE_LOG_DIR=/tmp/test-output \
    yolo-object-counter:0.1.0 \
    --image-dir /images --continuous N

# BioCLIP
docker run --rm --gpus all \
    -v ~/AI-projects/Sage-agents/plugins/bioclip-species-classifier/tests/test-images:/images:ro \
    -e PYWAGGLE_LOG_DIR=/tmp/test-output \
    bioclip-species-classifier:0.1.0 \
    --image-dir /images --continuous N

# vLLM — single image (launches vLLM server internally)
docker run --rm --gpus all --ipc=host \
    -v ~/AI-projects/Sage-agents/plugins/vllm-edge-inference/tests/test-images:/images:ro \
    -e PYWAGGLE_LOG_DIR=/tmp/test-output \
    vllm-edge-inference:0.1.0 \
    --stream /images/test-image001.jpg --continuous N \
    --enforce-eager --gpu-mem-frac 0.58
```


## 3. Push to Registry

### Option A: Sage ECR (production)

Tag for the Sage registry and push:

```bash
docker tag yolo-object-counter:0.1.0 \
    registry.sagecontinuum.org/flint-pete/yolo-object-counter:0.1.0
docker push registry.sagecontinuum.org/flint-pete/yolo-object-counter:0.1.0
```

Requires Sage portal credentials. See:
https://portal.sagecontinuum.org/account/access

### Option B: Docker save/load (no registry)

For testing without registry access, save the image to a file
and transfer it to the Thor node:

```bash
# On the build machine
docker save yolo-object-counter:0.1.0 | gzip > yolo-object-counter.tar.gz
scp yolo-object-counter.tar.gz user@thor-node:~/

# On the Thor node
sudo k3s ctr images import ~/yolo-object-counter.tar.gz
```

Note: `k3s ctr` (not `docker load`) because Thor nodes use
containerd under k3s, not standalone Docker.


## 4. Run on a Thor Node via pluginctl

SSH to the Thor node and use `pluginctl` (the official Sage
plugin testing tool). All `pluginctl` commands require `sudo`
because they interact with k3s:

```bash
ssh user@thor-node

# If using docker save/load (Option B), the image is already imported.
# If using Sage ECR (Option A), pluginctl pulls automatically.

# Build from source (if the node has internet — rare)
cd ~/sage-edge-plugins/plugins/yolo-object-counter
sudo pluginctl build .
# Output: 10.31.81.1:5000/local/yolo-object-counter

# Deploy (run the container)
sudo pluginctl deploy -n yolo-counter \
    10.31.81.1:5000/local/yolo-object-counter \
    -- --stream bottom_camera --continuous N

# Or deploy from the Sage registry
sudo pluginctl deploy -n yolo-counter \
    registry.sagecontinuum.org/flint-pete/yolo-object-counter:0.1.0 \
    -- --stream bottom_camera --continuous N

# Check status
sudo pluginctl ps

# View logs
pluginctl logs yolo-counter

# Stop the plugin (important — plugins run forever if not stopped)
sudo pluginctl rm yolo-counter
```

### Debugging inside the container

Override the entrypoint to get a shell:

```bash
sudo pluginctl deploy -n debug-yolo \
    --entrypoint /bin/bash \
    10.31.81.1:5000/local/yolo-object-counter \
    -- -c 'while true; do date; sleep 1; done'

# Attach to the running container
sudo pluginctl exec -ti debug-yolo -- /bin/bash

# Clean up when done
sudo pluginctl rm debug-yolo
```


## 5. Verify Data

After running, check the published data:

- **Sage Portal**: Data → Data Query Browser → filter by your plugin
- **sage-data-client** (Python):
  ```python
  import sage_data_client
  df = sage_data_client.query(
      start="-1h",
      filter={"name": "env.count.*", "plugin": "*yolo*"}
  )
  print(df[["timestamp", "name", "value"]].to_string())
  ```
- **Local output** (if testing with PYWAGGLE_LOG_DIR):
  ```bash
  cat /tmp/test-output/data.ndjson | python3 -m json.tool
  ls /tmp/test-output/uploads/
  ```


## Why Not Run Directly on the Node?

The Sage developer docs explicitly warn:

> ⚠️ Do not run any app or install packages directly on the node.
> Use Docker container or pluginctl commands.

Source: https://sagecontinuum.org/docs/reference-guides/dev-quick-reference

Reasons:
- GPU device permissions (`/dev/nvmap`) are configured for container
  access, not direct user access. Your user is likely not in the
  `video` group, so `torch.cuda.is_available()` returns False.
- Node environments are managed by the Sage team. Installing pip
  packages directly can break system services.
- Containers provide isolation and reproducibility — the same image
  runs identically on any Sage node.
- This matches the production deployment path: Docker image → ECR →
  edge scheduler → containerized execution.


## Troubleshooting

**pluginctl: dial tcp 10.31.81.1:6443 i/o timeout**
  → You forgot `sudo`. pluginctl needs root to read the k3s
  kubeconfig at /etc/rancher/k3s/k3s.yaml.

**pluginctl build: pip install fails with DNS resolution error**
  → The node has no outbound internet. Build on a machine with
  internet access and transfer the image (see Section 3, Option B).

**numpy.core.multiarray failed to import**
  → The Dockerfile's OpenCV fix didn't run. Rebuild with --no-cache:
  `docker build --no-cache -t <name> .`

**"No CUDA GPUs are available" inside container**
  → Missing --gpus all flag, or NVIDIA Container Toolkit not installed.
  Check: `docker run --rm --gpus all nvidia/cuda:12.0.0-base nvidia-smi`

**Image too large to transfer**
  → The vLLM image is ~80 GB. Use a fast network, or build directly
  on a machine with registry access and push to Sage ECR.
