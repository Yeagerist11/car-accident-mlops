from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/processed")

for split in ["training", "validation"]:
    print(f"\n{split.upper()} DATASET")
    print("-" * 35)

    counts = Counter()

    for class_folder in DATASET_DIR.joinpath(split).iterdir():
        if class_folder.is_dir():
            image_count = len([
                file for file in class_folder.iterdir()
                if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ])
            counts[class_folder.name] = image_count

    total = sum(counts.values())

    for class_name, count in sorted(counts.items()):
        percentage = (count / total) * 100 if total else 0
        print(f"{class_name}: {count} images ({percentage:.2f}%)")

    print(f"Total images: {total}")
    