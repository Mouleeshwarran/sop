from ultralytics import YOLO
import time

MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_PATH = "testing/videos/Ob1.mp4"

model = YOLO(MODEL_PATH)

SKIP = 2

print("Starting frame-skip benchmark...")
print(f"Processing every {SKIP}nd frame")

start = time.time()

total_frames = 0
processed_frames = 0

for result in model.predict(
    source=VIDEO_PATH,
    imgsz=512,
    conf=0.25,
    stream=True,
    vid_stride=SKIP,
    verbose=False
):
    processed_frames += 1

elapsed = time.time() - start

# Original video frame count
import cv2

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

fps = processed_frames / elapsed

print("\n" + "=" * 50)
print("FRAME SKIP BENCHMARK")
print("=" * 50)
print(f"Original frames  : {total_frames}")
print(f"Processed frames : {processed_frames}")
print(f"Total time       : {elapsed:.2f} seconds")
print(f"Processing FPS   : {fps:.2f}")
print(f"Time per frame   : {(elapsed / processed_frames) * 1000:.1f} ms")
print("=" * 50)