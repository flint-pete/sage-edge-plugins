# YOLO Object Counter — Sage/Waggle edge plugin
# Default model: YOLO11x (54.7% mAP COCO, 56.9M params)
# Target: 128GB unified memory ARM64 (DGX Spark / Sage Thor)
FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-download default model weights into the image
RUN python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"

ENTRYPOINT ["python", "/app/app.py"]
