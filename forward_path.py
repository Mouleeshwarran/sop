import cv2
from ultralytics import YOLO
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

INPUT_VIDEO = "testing/videos/pothole_video.mp4"

OUTPUT_DIR = Path("testing/results/forward_path")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "pothole_forward_path.mp4"

CONFIDENCE = 0.25
IMAGE_SIZE = 640


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("ERROR: Could not open video!")
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
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# FORWARD PATH REGION
# ============================================================

# Horizontal boundaries
path_left = int(width * 0.30)
path_right = int(width * 0.70)

# Vertical boundaries
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
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    pothole_in_path = False
    pothole_count = 0

    # --------------------------------------------------------
    # DRAW FORWARD PATH
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Our dataset:
            # 0 = crocodile crack
            # 1 = longitudinal crack
            # 2 = pothole

            if class_id != 2:
                continue

            pothole_count += 1

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Center of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # ------------------------------------------------
            # CHECK WHETHER POTHOLE IS IN FORWARD PATH
            # ------------------------------------------------

            inside_path = (
                path_left <= center_x <= path_right
                and
                path_top <= center_y <= path_bottom
            )

            if inside_path:
                pothole_in_path = True

                box_color = (0, 0, 255)

            else:
                box_color = (0, 165, 255)

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # Label
            label = f"Pothole {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                box_color,
                -1
            )

    # ========================================================
    # STATUS
    # ========================================================

    if pothole_in_path:

        status = "WARNING: POTHOLE AHEAD"
        status_color = (0, 0, 255)

    elif pothole_count > 0:

        status = "POTHOLE DETECTED - OUTSIDE PATH"
        status_color = (0, 165, 255)

    else:

        status = "PATH CLEAR"
        status_color = (0, 255, 0)

    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (width - 10, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        status,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    # Frame counter
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
    # SAVE FRAME
    # ========================================================

    out.write(frame)

    # Progress
    if frame_number % 25 == 0:

        progress = (frame_number / total_frames) * 100

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
print("FORWARD PATH ANALYSIS COMPLETED")
print("=" * 60)

print(f"Output video:")
print(OUTPUT_VIDEO)