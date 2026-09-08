from ultralytics import YOLO
from pathlib import Path

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = YOLO("runs/detect/train/weights/best.pt")


# ==============================
# VIDEO PATHS
# ==============================

videos = {
    "Pothole Video": "testing/videos/pothole_video.mp4",
    "Clean Video": "testing/videos/clean_video.mp4"
}


# ==============================
# OUTPUT FOLDER
# ==============================

output_folder = Path("testing/results")
output_folder.mkdir(parents=True, exist_ok=True)


# ==============================
# PROCESS VIDEOS
# ==============================

for name, video_path in videos.items():

    print("\n" + "=" * 60)
    print(f"Processing: {name}")
    print(f"Video: {video_path}")
    print("=" * 60)

    if not Path(video_path).exists():
        print(f"ERROR: Video not found -> {video_path}")
        continue

    # Choose output filename
    if name == "Pothole Video":
        output_name = "pothole_result.mp4"
    else:
        output_name = "clean_result.mp4"

    output_path = output_folder / output_name

    # Run YOLO detection
    results = model.predict(
        source=video_path,
        conf=0.25,
        imgsz=640,
        save=True,
        project=str(output_folder),
        name=name.lower().replace(" ", "_"),
        exist_ok=True
    )

    print(f"\nCompleted: {name}")
    print(f"Result saved in YOLO output folder.")


print("\n" + "=" * 60)
print("ALL VIDEO TESTS COMPLETED")
print("=" * 60)
print("Check:")
print("testing/results/")