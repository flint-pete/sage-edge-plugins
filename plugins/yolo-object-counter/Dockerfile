# YOLO Object Counter — Sage/Waggle edge plugin
# Default model: YOLO11x (54.7% mAP COCO, 56.9M params)
# Target: 128GB unified memory ARM64 (DGX Spark / Sage Thor)
FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Fix OpenCV headless conflict with pre-installed version in NVIDIA base image
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless>=4.8.0

# Pre-download default model weights into the image
# Layer order matters: model weights change rarely, app.py changes often.
# Putting the model download BEFORE COPY app.py means code edits don't
# invalidate the expensive (~130 MB) model download layer.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"

COPY app.py .

ENTRYPOINT ["python", "/app/app.py"]
