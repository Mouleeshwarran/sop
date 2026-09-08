from ultralytics import YOLO
from pathlib import Path

# Load our trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Images to test
images = {
    "Pothole Road": "pothole.jpeg",
    "Clean Road": "clean.jpeg"
}

# Run detection on both images
for name, image_path in images.items():

    print(f"\nProcessing: {name}")
    print(f"Image: {image_path}")

    # Check if image exists
    if not Path(image_path).exists():
        print(f"ERROR: {image_path} not found!")
        continue

    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=640,
        save=True
    )

    # Display detection information
    for result in results:

        if len(result.boxes) == 0:
            print("  No pothole detected.")
        else:
            print(f"  {len(result.boxes)} pothole(s) detected!")

            for box in result.boxes:
                confidence = float(box.conf[0])
                print(f"  Confidence: {confidence:.2f}")

print("\n================================")
print("Detection completed!")
print("================================")
print("Check the 'runs/detect/' folder for results.")