# Testing Plugins on Thor Nodes

How to test Sage edge plugins on a Thor node. There are two
approaches, depending on whether you use Docker (recommended)
or run directly.


## Approach 1: Docker / pluginctl (Recommended)

This is the officially supported Sage workflow. Plugins run inside
Docker containers, which have proper GPU access. The Sage developer
docs explicitly state:

> ⚠️ Do not run any app or install packages directly on the node.
> Use Docker container or pluginctl commands.
> — https://sagecontinuum.org/docs/reference-guides/dev-quick-reference

### Quick start

Build on a machine with internet (DGX Spark), transfer to Thor,
run via pluginctl. Full instructions in **DOCKER-BUILD.md**.

```bash
# === On the build machine (DGX Spark) ===
cd ~/AI-projects/Sage-agents/plugins/yolo-object-counter
docker build --no-cache -t yolo-object-counter:0.1.0 .

# Save and transfer to Thor
docker save yolo-object-counter:0.1.0 | gzip > /tmp/yolo.tar.gz
scp /tmp/yolo.tar.gz user@thor-node:~/

# === On the Thor node ===
sudo k3s ctr images import ~/yolo.tar.gz
sudo pluginctl deploy -n yolo-counter \
    docker.io/library/yolo-object-counter:0.1.0 \
    -- --stream bottom_camera --continuous N
pluginctl logs yolo-counter
sudo pluginctl rm yolo-counter
```

### Why this approach?

- GPU access works out of the box (containers get `/dev/nvmap` access)
- Matches the production deployment path (Docker → ECR → scheduler)
- No risk of polluting the node's system packages
- Reproducible across nodes

See **DOCKER-BUILD.md** for complete build, push, and deploy
instructions.


## Approach 2: Direct Execution (Requires Admin Setup)

Running `python3 app.py` directly on the node is faster for
iteration but requires your user to be in the `video` group for
GPU access. Without this, `torch.cuda.is_available()` returns
False due to `/dev/nvmap` permission restrictions.

### GPU access check

```bash
# Check if your user has GPU access
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, check /dev/nvmap permissions
ls -la /dev/nvmap
# cr--r----- 1 root video ... /dev/nvmap  ← video group required

# Check your groups
groups
# If 'video' is not listed, ask your admin:
# sudo usermod -aG video <username>
# Then log out and back in
```

### Setup (if you have GPU access)

```bash
ssh user@thor-node
cd ~/sage-edge-plugins

# Clone or pull the latest code
git pull  # or: git clone https://github.com/flint-pete/sage-edge-plugins.git

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies (all three plugins share one venv)
pip install -r plugins/yolo-object-counter/requirements.txt
pip install -r plugins/bioclip-species-classifier/requirements.txt
pip install -r plugins/vllm-edge-inference/requirements.txt
```

### Running YOLO

```bash
source venv/bin/activate
export PYWAGGLE_LOG_DIR=./output/yolo-run

# Batch mode — all test images
python3 plugins/yolo-object-counter/app.py \
  --image-dir plugins/yolo-object-counter/tests/test-images \
  --continuous N

# Check results
cat output/yolo-run/data.ndjson | python3 -m json.tool --no-ensure-ascii
ls output/yolo-run/uploads/
```

Key arguments:
```
--image-dir DIR          Process all images in a directory
--continuous N           Run once and exit (don't loop)
--model yolo11x.pt       Model file (auto-downloads ~130MB on first run)
--conf-thres 0.25        Confidence threshold (lower = more detections)
--iou-thres 0.45         NMS IoU threshold
--imgsz 640              Input resolution (larger = better small-object
                         detection, more GPU memory)
--half                   FP16 inference (faster, slightly less accurate)
--max-det 300            Maximum detections per image
--augment                Test-time augmentation (~3x slower, more accurate)
--agnostic-nms           Class-agnostic NMS
--classes "person,car"   Filter to specific COCO classes
--upload-image Y         Save annotated images to uploads/
```

See: https://docs.ultralytics.com/modes/predict/#inference-arguments

### Running BioCLIP

```bash
export PYWAGGLE_LOG_DIR=./output/bioclip-run

python3 plugins/bioclip-species-classifier/app.py \
  --image-dir plugins/bioclip-species-classifier/tests/test-images \
  --continuous N

cat output/bioclip-run/data.ndjson | python3 -m json.tool --no-ensure-ascii
```

Key arguments:
```
--image-dir DIR          Process all images in a directory
--continuous N           Run once and exit
--rank Species           Taxonomy rank: Kingdom, Phylum, Class, Order,
                         Family, Genus, or Species (default: Class)
--model hf-hub:imageomics/bioclip-2   BioCLIP model (auto-downloads ~1.5GB)
--min-confidence 0.1     Minimum confidence to report
--top-k 5                Number of top predictions to publish
```

### Running vLLM

```bash
export PYWAGGLE_LOG_DIR=./output/vllm-run

# Single image (vLLM has no --image-dir)
python3 plugins/vllm-edge-inference/app.py \
  --stream plugins/vllm-edge-inference/tests/test-images/test-image001.jpg \
  --continuous N \
  --enforce-eager \
  --gpu-mem-frac 0.58

cat output/vllm-run/data.ndjson | python3 -m json.tool --no-ensure-ascii
```

Key arguments:
```
--stream IMAGE           Camera, RTSP URL, or path to a single image
--continuous N           Process one frame and exit
--model Qwen/Qwen3-VL-32B-Instruct   HuggingFace model (~67GB download)
--gpu-mem-frac 0.58      GPU memory fraction (0.58 for 128GB unified)
--enforce-eager          Skip CUDA graph capture (prevents OOM)
--max-model-len 4096     Max sequence length
--dtype bfloat16         Data type (bfloat16 for Blackwell)
--max-tokens 512         Max tokens in the response
--temperature 0.3        Generation temperature
--vllm-port 8000         vLLM server port (change if 8000 is in use)
--upload-image Y         Save source image to uploads/
```

**Unified memory notes (Thor / DGX Spark):**
- Always use `--enforce-eager` (CUDA graph capture OOMs on 32B+ models)
- Keep `--gpu-mem-frac` at 0.55–0.60 (unified memory shared with OS)
- vLLM server takes 2–5 minutes to load the model — be patient
- Check for conflicts: `nvidia-smi` and `ps aux | grep vllm`

### Running the test suites

```bash
source venv/bin/activate

# YOLO — detailed per-image report
python3 plugins/yolo-object-counter/tests/test_yolo_local.py

# BioCLIP — species classification report
python3 plugins/bioclip-species-classifier/tests/test_bioclip_local.py

# vLLM — starts server, runs inference, validates output
python3 plugins/vllm-edge-inference/tests/test_vllm_integration.py

# Run all three
bash tests/run-all-tests.sh
```


## Inspecting Output

All output is captured as NDJSON (newline-delimited JSON):

```bash
# Pretty-print all published measurements
cat output/yolo-run/data.ndjson | python3 -m json.tool

# View uploaded images
ls -la output/yolo-run/uploads/

# Copy back to your dev machine for viewing
scp user@thor-node:~/sage-edge-plugins/output/yolo-run/uploads/* ./
```


## Clean Up

```bash
# Remove all test outputs, model weights, caches, macOS junk
./clean.sh --force

# Or manually:
rm -rf output/
rm -f yolo11x.pt                    # YOLO model (~130MB)
rm -rf ~/.cache/huggingface/hub/    # All HuggingFace models
deactivate
```


## Troubleshooting

**torch.cuda.is_available() returns False**
  → Your user is not in the `video` group. The `/dev/nvmap` device
  is owned by `root:video`. Ask your admin to run:
  `sudo usermod -aG video <username>`
  Then log out and back in.

**NvRmMemInitNvmap failed: Permission denied**
  → Same cause as above — `/dev/nvmap` not accessible. These are
  Tegra driver messages, not standard CUDA errors.

**pluginctl: dial tcp 10.31.81.1:6443 i/o timeout**
  → You forgot `sudo`. pluginctl needs root to read the k3s
  kubeconfig.

**pluginctl build: pip install fails with DNS error**
  → The node has no outbound internet. Build the Docker image on a
  machine with internet and transfer it. See DOCKER-BUILD.md.

**YOLO model downloads on first run**
  → Normal for direct execution. Ultralytics auto-downloads
  yolo11x.pt (~130MB) to the working directory. Subsequent runs
  use the cached file. In Docker, the model is baked into the image.

**BioCLIP hangs on first run**
  → Downloading the model from HuggingFace (~1.5GB). Check progress:
  `du -sh ~/.cache/huggingface/`

**vLLM OOM (Out of Memory)**
  → Lower `--gpu-mem-frac` (try 0.50) and ensure `--enforce-eager`
  is set. Check for other GPU processes: `nvidia-smi`

**vLLM server never becomes ready**
  → Port conflict: `ss -tlnp | grep 8000`
  → Zombie process: `ps aux | grep vllm`
  → Try a different port: `--vllm-port 8199`
