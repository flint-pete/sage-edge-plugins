# Testing Plugins Directly on a Thor Node (No Docker)

How to copy the plugin code to a Thor node, install dependencies,
and run the plugins directly from the command line — bypassing the
Docker build / ECR submission pipeline entirely.

This is the fastest way to iterate on a plugin before committing
to a Docker image.


## Prerequisites

- SSH access to a Thor node (you'll need the hostname or IP)
- The Thor node has an NVIDIA GPU with CUDA and Python 3.10+
- A few test images (JPG/PNG) to feed the plugins


## 1. Upload the Code

From your development machine (DGX Spark), push the repo to the
Thor node. Adjust `THOR` to your node's address:

```bash
export THOR=user@thor-node-hostname

# Option A: rsync the whole repo (fastest for first transfer)
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'tests/.venv' --exclude '*.pt' --exclude '*.safetensors' \
  ~/AI-projects/Sage-agents/ $THOR:~/sage-edge-plugins/

# Option B: git clone (if the Thor node has internet access)
ssh $THOR "git clone https://github.com/flint-pete/sage-edge-plugins.git"
```


## 2. Set Up a Python Virtual Environment (on Thor)

SSH into the node and create a venv. All three plugins can share
one venv since their dependencies are compatible:

```bash
ssh $THOR
cd ~/sage-edge-plugins

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```


## 3. Install Plugin Dependencies

Install each plugin's requirements. Order matters — install vLLM
last since it's the heaviest and may upgrade torch:

```bash
# YOLO
pip install -r plugins/yolo-object-counter/requirements.txt

# BioCLIP
pip install -r plugins/bioclip-species-classifier/requirements.txt

# vLLM (heavyweight — installs vllm + torch if not already present)
pip install -r plugins/vllm-edge-inference/requirements.txt
```

All three plugins require `pywaggle`, which is in each requirements.txt.

If you prefer isolated per-plugin venvs, create separate ones:

```bash
cd plugins/yolo-object-counter
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
# ... test ... deactivate
```


## 4. Test Images

Each plugin ships with test images in its `tests/test-images/`
directory (committed to the repo). These travel with the code
and are ready to use — no generation step needed.

To add your own test images, drop JPG/PNG files into the
plugin's test-images directory:

```bash
cp my-photo.jpg plugins/yolo-object-counter/tests/test-images/
cp my-photo.jpg plugins/bioclip-species-classifier/tests/test-images/
cp my-photo.jpg plugins/vllm-edge-inference/tests/test-images/
```


## 5. Run Each Plugin

All three plugins use PYWAGGLE_LOG_DIR to capture output locally
instead of publishing to a real Sage Beehive. The published
measurements go to `data.ndjson` and uploaded files go to
`uploads/` inside the log directory.


### 5a. YOLO Object Counter

```bash
cd ~/sage-edge-plugins
source venv/bin/activate

export PYWAGGLE_LOG_DIR=./output/yolo-run

# Batch mode — process all images in a directory
python3 plugins/yolo-object-counter/app.py \
  --image-dir plugins/yolo-object-counter/tests/test-images \
  --continuous N

# Check results
cat output/yolo-run/data.ndjson | python3 -m json.tool --no-ensure-ascii
ls output/yolo-run/uploads/
```

Key arguments:
  --image-dir DIR      Process all images in a directory (batch mode)
  --continuous N       Run once and exit (don't loop)
  --model yolo11x.pt   Model file (auto-downloads on first run, ~130MB)
  --conf-thres 0.25    Confidence threshold (lower = more detections)
  --iou-thres 0.45     NMS IoU threshold
  --classes "person,car,bird"   Filter to specific COCO classes
  --upload-image Y     Save annotated images to uploads/

The first run downloads yolo11x.pt (~130MB) automatically.

To test with a single image instead of a directory:

```bash
python3 plugins/yolo-object-counter/app.py \
  --stream plugins/yolo-object-counter/tests/test-images/test-image001.jpg \
  --continuous N
```


### 5b. BioCLIP Species Classifier

```bash
cd ~/sage-edge-plugins
source venv/bin/activate

export PYWAGGLE_LOG_DIR=./output/bioclip-run

python3 plugins/bioclip-species-classifier/app.py \
  --image-dir plugins/bioclip-species-classifier/tests/test-images \
  --continuous N

cat output/bioclip-run/data.ndjson | python3 -m json.tool --no-ensure-ascii
```

Key arguments:
  --image-dir DIR      Process all images in a directory
  --continuous N       Run once and exit
  --rank Species       Taxonomy rank: Kingdom, Phylum, Class, Order,
                       Family, Genus, or Species (default: Class)
  --model hf-hub:imageomics/bioclip-2   BioCLIP model (auto-downloads)
  --min-confidence 0.1  Minimum confidence to report
  --top-k 5            Number of top predictions to publish

The first run downloads BioCLIP-2 from HuggingFace (~1.5GB).
Subsequent runs use the cached model.

To test with a single image:

```bash
python3 plugins/bioclip-species-classifier/app.py \
  --stream plugins/bioclip-species-classifier/tests/test-images/test-image001.jpg \
  --rank Species \
  --continuous N
```


### 5c. vLLM Edge Inference (Scene Description)

This plugin is different from the others — it launches a vLLM
server as a subprocess, waits for it to load the model, then
sends camera frames to it for scene description. It does NOT
have --image-dir, but you can point --stream at a single image.

**Warning**: The default model (Qwen3-VL-32B-Instruct) is ~67GB.
First download takes 10-15 minutes. Make sure you have disk space.

```bash
cd ~/sage-edge-plugins
source venv/bin/activate

export PYWAGGLE_LOG_DIR=./output/vllm-run

# Single image, one-shot (uses a shipped test image)
python3 plugins/vllm-edge-inference/app.py \
  --stream plugins/vllm-edge-inference/tests/test-images/test-image001.jpg \
  --continuous N \
  --enforce-eager \
  --gpu-mem-frac 0.58

cat output/vllm-run/data.ndjson | python3 -m json.tool --no-ensure-ascii
```

Key arguments:
  --stream IMAGE       Camera stream, RTSP URL, or path to a single image
  --continuous N       Process one frame and exit
  --model Qwen/Qwen3-VL-32B-Instruct   HuggingFace model name
  --gpu-mem-frac 0.58  Fraction of GPU memory for vLLM (0.58 for 128GB
                       unified memory — do NOT use 0.8+, it will OOM)
  --enforce-eager      Skip CUDA graph capture (saves ~5GB, prevents OOM
                       on unified memory with large models)
  --max-model-len 4096 Max sequence length
  --dtype bfloat16     Data type (bfloat16 recommended for Blackwell)
  --max-tokens 512     Max tokens in the response
  --temperature 0.3    Generation temperature
  --prompt "..."       Custom description prompt
  --summary-prompt "..." Custom summary prompt
  --vllm-port 8000     Port for the vLLM server (change if 8000 is in use)
  --upload-image Y     Save source image to uploads/

**Important notes for unified memory nodes (Thor / DGX Spark):**
- Always use --enforce-eager (CUDA graph capture OOMs on 32B+ models)
- Keep --gpu-mem-frac at 0.55-0.60 (unified memory is shared with OS)
- The vLLM server takes 2-5 minutes to load the model — be patient
- If it hangs or crashes, check if another vLLM process is using the GPU:
  `nvidia-smi` and `ps aux | grep vllm`

To run multiple images through vLLM, process them in a loop:

```bash
for img in plugins/vllm-edge-inference/tests/test-images/*.jpg; do
  export PYWAGGLE_LOG_DIR=./output/vllm-run
  python3 plugins/vllm-edge-inference/app.py \
    --stream "$img" \
    --continuous N \
    --enforce-eager \
    --gpu-mem-frac 0.58
done
```

(Each invocation restarts the vLLM server, which is slow. For
batch testing, consider using the integration test instead — see
Section 7 below.)


## 6. Inspect the Output

All output is captured as NDJSON (newline-delimited JSON):

```bash
# Pretty-print all published measurements
cat output/yolo-run/data.ndjson | python3 -c "
import sys, json
for line in sys.stdin:
    obj = json.loads(line)
    print(json.dumps(obj, indent=2))
"

# Count detections per class (YOLO)
cat output/yolo-run/data.ndjson | python3 -c "
import sys, json
for line in sys.stdin:
    obj = json.loads(line)
    print(f\"{obj['name']:30s}  value={obj['value']}  meta={obj.get('meta',{})}\")
"

# View uploaded images
ls -la output/yolo-run/uploads/
# Open with any image viewer, or scp back to your dev machine:
# scp $THOR:~/sage-edge-plugins/output/yolo-run/uploads/* ./
```


## 7. Running the Integration Tests

Each plugin has integration tests that exercise the full pipeline
(real models, real GPU inference) with better reporting:

```bash
# YOLO integration test
cd plugins/yolo-object-counter
python3 tests/test_yolo_local.py

# BioCLIP integration test
cd plugins/bioclip-species-classifier
python3 tests/test_bioclip_local.py

# vLLM integration test (starts vLLM server, runs multiple images)
cd plugins/vllm-edge-inference
python3 tests/test_vllm_integration.py
```

The integration tests produce detailed reports with per-image
results, timing, and confidence scores. They also validate that
pywaggle output is well-formed.

Put real test images in each plugin's test-images directory:
  plugins/yolo-object-counter/tests/test-images/
  plugins/bioclip-species-classifier/tests/test-images/
  plugins/vllm-edge-inference/tests/test-images/


## 8. Clean Up

```bash
# Remove captured output
rm -rf output/

# Remove downloaded models (if you want to free disk space)
rm -f yolo11x.pt                           # YOLO model (~130MB)
rm -rf ~/.cache/huggingface/hub/            # All HuggingFace models
# WARNING: this removes ALL cached HF models — BioCLIP + Qwen3

# Deactivate venv
deactivate
```


## Troubleshooting

**"ModuleNotFoundError: No module named 'waggle'"**
  → You forgot to activate the venv, or pywaggle isn't installed.
  Run: source venv/bin/activate && pip install pywaggle

**YOLO model downloads to current directory**
  → Normal. ultralytics puts yolo11x.pt in the working directory.
  The .gitignore already excludes *.pt files.

**BioCLIP hangs on first run**
  → It's downloading the model from HuggingFace (~1.5GB).
  Check progress with: du -sh ~/.cache/huggingface/

**vLLM OOM (Out of Memory)**
  → Lower --gpu-mem-frac (try 0.50) and make sure --enforce-eager
  is set. Check nvidia-smi for other processes using GPU memory.

**vLLM server never becomes ready**
  → Check if the port is in use: ss -tlnp | grep 8000
  → Check for zombie processes: ps aux | grep vllm
  → Try a different port: --vllm-port 8199

**"CUDA out of memory" during model load**
  → Another process may be using the GPU. Kill it first:
  nvidia-smi  (find the PID)
  kill <PID>

**Permission denied on SSH**
  → You need SSH access provisioned for the Thor node.
  Contact the Sage team for access credentials.
