import cv2
from ultralytics import YOLO
from pathlib import Path

# ============================================================
# MODELS
# ============================================================

POTHOLE_MODEL = "runs/detect/train/weights/best.pt"
OBJECT_MODEL = "yolo11n.pt"

# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_VIDEO = "testing/videos/Combined1.mp4"

OUTPUT_DIR = Path("testing/results/combined")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "combined_navigation.mp4"

# ============================================================
# SETTINGS
# ============================================================

POTHOLE_CONF = 0.25
OBJECT_CONF = 0.35
IMG_SIZE = 640

# Relevant road/path obstacles from COCO YOLO
RELEVANT_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
}

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading pothole model...")
pothole_model = YOLO(POTHOLE_MODEL)

print("Loading object model...")
object_model = YOLO(OBJECT_MODEL)

print("Both models loaded successfully!")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo information:")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")

# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# FORWARD PATH
# ============================================================

path_left = int(width * 0.30)
path_right = int(width * 0.70)

path_top = int(height * 0.45)
path_bottom = int(height * 0.95)

# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # POTHOLE DETECTION
    # --------------------------------------------------------

    pothole_results = pothole_model.predict(
        source=frame,
        conf=POTHOLE_CONF,
        imgsz=IMG_SIZE,
        verbose=False
    )

    # --------------------------------------------------------
    # OBJECT DETECTION
    # --------------------------------------------------------

    object_results = object_model.predict(
        source=frame,
        conf=OBJECT_CONF,
        imgsz=IMG_SIZE,
        verbose=False
    )

    pothole_in_path = False
    object_in_path = False

    hazard_names = []

    # ========================================================
    # DRAW FORWARD PATH
    # ========================================================

    cv2.rectangle(
        frame,
        (path_left, path_top),
        (path_right, path_bottom),
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "FORWARD PATH",
        (path_left + 10, path_top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ========================================================
    # PROCESS POTHOLES
    # ========================================================

    for result in pothole_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            # Class 2 = pothole
            if class_id != 2:
                continue

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            inside_path = (
                path_left <= center_x <= path_right
                and
                path_top <= center_y <= path_bottom
            )

            if inside_path:

                pothole_in_path = True
                hazard_names.append("POTHOLE")

                box_color = (0, 0, 255)

            else:

                box_color = (0, 165, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            cv2.putText(
                frame,
                f"Pothole {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                box_color,
                -1
            )

    # ========================================================
    # PROCESS OBJECTS
    # ========================================================

    for result in object_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = object_model.names[class_id]

            # Ignore irrelevant COCO classes
            if object_name not in RELEVANT_OBJECTS:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            inside_path = (
                path_left <= center_x <= path_right
                and
                path_top <= center_y <= path_bottom
            )

            if inside_path:

                object_in_path = True
                hazard_names.append(object_name.upper())

                box_color = (0, 0, 255)

            else:

                box_color = (0, 255, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            cv2.putText(
                frame,
                f"{object_name} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                box_color,
                -1
            )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    if pothole_in_path and object_in_path:

        status = "WARNING: POTHOLE + OBSTACLE AHEAD"
        status_color = (0, 0, 255)

    elif pothole_in_path:

        status = "WARNING: POTHOLE AHEAD"
        status_color = (0, 0, 255)

    elif object_in_path:

        status = "WARNING: OBSTACLE AHEAD"
        status_color = (0, 0, 255)

    else:

        status = "PATH CLEAR"
        status_color = (0, 255, 0)

    # ========================================================
    # STATUS BAR
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (width - 10, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        status,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    # ========================================================
    # FRAME COUNTER
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # ========================================================
    # SAVE
    # ========================================================

    out.write(frame)

    if frame_number % 25 == 0:

        progress = frame_number / total_frames * 100

        print(
            f"Processing: {progress:.1f}% "
            f"({frame_number}/{total_frames})"
        )

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

print("\n" + "=" * 60)
print("COMBINED NAVIGATION ANALYSIS COMPLETED")
print("=" * 60)

print(f"Output:")
print(OUTPUT_VIDEO)