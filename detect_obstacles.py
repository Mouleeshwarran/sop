from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run on our pothole video
results = model.predict(
    source="testing/videos/Ob1.mp4",
    conf=0.35,
    imgsz=640,
    save=True
)

print("\nObstacle detection completed!")
print("Check the runs/detect/ folder for the output video.")