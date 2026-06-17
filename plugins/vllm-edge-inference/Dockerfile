# vLLM Edge Inference — Sage/Waggle edge plugin
# Single-stage build: vLLM server + plugin application
# Target: Sage Thor nodes (128 GB unified memory, Blackwell GPU)
#
# Base image: NVIDIA PyTorch 25.04 — CUDA 12.9, PyTorch 2.7, Python 3.12
# Supports Blackwell GPUs (sm_120/sm_121) natively. Requires driver R575+.
FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app

# Install vLLM and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# The NVIDIA base image may ship an opencv compiled against a different numpy.
# Fix: fully remove old opencv (pip uninstall + rm stale files), then
# install a fresh opencv-python-headless matching the current numpy.
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2* && \
    pip install --no-cache-dir opencv-python-headless>=4.8.0

# Pre-download default VLM weights into the image
# Layer order matters: model weights change rarely, app.py changes often.
# Putting the model download BEFORE COPY app.py means code edits don't
# invalidate the expensive (~67 GB) model download layer.
# Qwen3-VL-32B-Instruct is ~67GB BF16 — designed for 128GB nodes (Thor / DGX Spark)
RUN mkdir -p /hf_cache && \
    HF_HOME=/hf_cache huggingface-cli download \
    Qwen/Qwen3-VL-32B-Instruct \
    --cache-dir /hf_cache \
    --resume

ENV HF_HOME=/hf_cache

COPY app.py .

ENTRYPOINT ["python3", "/app/app.py"]
