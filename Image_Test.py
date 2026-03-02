import cv2
import numpy as np
from ultralytics import YOLO

# ======================================
# CONFIG
# ======================================
MODEL_PATH = "best_ncnn_model"
IMAGE_PATH = "test (5).jpg"   # <-- your image

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 416

# ======================================
# LOAD MODEL
# ======================================
model = YOLO(MODEL_PATH, task="segment")
print("Model Loaded:", MODEL_PATH)

# ======================================
# LOAD IMAGE
# ======================================
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise Exception("Image not found")

h, w = image.shape[:2]

# ======================================
# INFERENCE
# ======================================
results = model(
    image,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    imgsz=IMG_SIZE,
    retina_masks=True,
    verbose=False
)[0]

print("\n==============================")
print("RAW TENSOR INFORMATION")
print("==============================")

# --------------------------------------
# BOXES
# --------------------------------------
if results.boxes is not None:
    print("Boxes tensor shape:", results.boxes.data.shape)
else:
    print("No boxes detected")

# --------------------------------------
# MASKS
# --------------------------------------
if results.masks is not None:
    print("Masks tensor shape:", results.masks.data.shape)
else:
    print("No masks detected")

print("\n==============================")
print("PER OBJECT DETAILS")
print("==============================")

vis = image.copy()

if results.boxes is not None:

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)

    polygons = results.masks.xy if results.masks is not None else []

    for i in range(len(boxes)):

        x1, y1, x2, y2 = boxes[i]
        conf = confs[i]
        cls = clss[i]

        # convert to xywh
        bw = x2 - x1
        bh = y2 - y1

        print(f"\n--- OBJECT {i} ---")
        print(f"Class ID      : {cls}")
        print(f"Class Name    : {model.names[cls]}")
        print(f"Confidence    : {conf:.4f}")
        print(f"x,y,w,h       : ({x1:.1f}, {y1:.1f}, {bw:.1f}, {bh:.1f})")

        # ----------------------------------
        # MASK POLYGON POINTS
        # ----------------------------------
        if len(polygons) > i:
            pts = polygons[i]
            print("Mask points shape:", pts.shape)
            print("First 10 points:")
            print(pts[:10])

            # draw polygon
            pts_int = pts.astype(np.int32)
            cv2.polylines(vis, [pts_int], True, (0,255,0), 2)

        # draw bbox
        cv2.rectangle(
            vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255,0,0),
            2
        )

        cv2.putText(
            vis,
            f"{model.names[cls]} {conf:.2f}",
            (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

# ======================================
# SHOW RESULT
# ======================================
cv2.imshow("Image Debug", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()