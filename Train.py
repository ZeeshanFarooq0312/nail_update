
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # or try yolov11n-seg.pt / newer nano if available

model.train(
    task="segment",
    data="data.yaml",
    epochs=100,                  # More room for mask convergence
    imgsz=640,                   # ↑ from 320 — better nail detail without killing speed
    batch=16,
    device=0,

    overlap_mask=True,
    mask_ratio=4,                # Critical: faster + more stable masks on mobile

    box=7.5,
    cls=0.5,
    dfl=1.5,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=12,                  # Slightly gentler
    translate=0.1,
    scale=0.3,
    shear=0,                     # ← Disable for natural nail shapes
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.3,
    mixup=0.0,
    copy_paste=0.05,             # Lower to reduce artifacts

    dropout=0.0,

    amp=True,                    # Faster training + lower memory
    
    # name="yolov8n-seg_mobile416_v27",  # Update name for tracking
    exist_ok=True
)