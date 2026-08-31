from pathlib import Path
from collections import Counter

DATASET = Path("dataset/pothole_dataset")

splits = {
    "train": DATASET / "train",
    "valid": DATASET / "valid",
    "test": DATASET / "test",
}

class_names = {
    0: "crocodile crack",
    1: "longitudinal crack",
    2: "pothole",
}

print("=" * 60)
print("YOLO DATASET HEALTH CHECK")
print("=" * 60)

total_images = 0
total_labels = 0
total_boxes = 0
total_invalid = 0

for split_name, split_path in splits.items():

    image_dir = split_path / "images"
    label_dir = split_path / "labels"

    images = list(image_dir.glob("*"))
    images = [
        p for p in images
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    class_counter = Counter()
    boxes = 0
    invalid = 0

    for label_file in labels:
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    invalid += 1
                    continue

                class_id = int(parts[0])
                values = list(map(float, parts[1:]))

                if class_id not in class_names:
                    invalid += 1
                    continue

                # YOLO format:
                # class x_center y_center width height
                if not all(0 <= v <= 1 for v in values):
                    invalid += 1
                    continue

                class_counter[class_id] += 1
                boxes += 1

        except Exception:
            invalid += 1

    print(f"\n[{split_name.upper()}]")
    print(f"Images          : {len(images)}")
    print(f"Labels          : {len(labels)}")
    print(f"Bounding boxes  : {boxes}")
    print(f"Missing labels  : {len(missing_labels)}")
    print(f"Orphan labels   : {len(orphan_labels)}")
    print(f"Invalid labels  : {invalid}")

    print("Class distribution:")

    for class_id, name in class_names.items():
        print(f"  {class_id} - {name}: {class_counter[class_id]}")

    total_images += len(images)
    total_labels += len(labels)
    total_boxes += boxes
    total_invalid += invalid

print("\n" + "=" * 60)
print("TOTAL")
print("=" * 60)

print(f"Images          : {total_images}")
print(f"Labels          : {total_labels}")
print(f"Bounding boxes  : {total_boxes}")
print(f"Invalid labels  : {total_invalid}")

if total_invalid == 0:
    print("\nSTATUS: ✅ Dataset looks healthy.")
else:
    print("\nSTATUS: ⚠️ Dataset has invalid annotations.")

print("=" * 60)