# vLLM Edge Inference — Sage/Waggle edge plugin
# Single-stage build: vLLM server + plugin application
# Target: Sage Thor nodes (128 GB unified memory, Blackwell GPU)
FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /app

# Install vLLM and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# The NVIDIA base image ships opencv 4.7 compiled against numpy 1.x.
# vllm/transformers pull numpy 2.x, which is binary-incompatible.
# Fix: fully remove old opencv (pip uninstall + rm stale files), then
# install a fresh opencv-python-headless built against numpy 2.x.
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2* && \
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

ENTRYPOINT ["python", "/app/app.py"]
