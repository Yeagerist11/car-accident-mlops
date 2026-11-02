
from pathlib import Path
import hashlib
import shutil
import csv

# Original dataset location
SOURCE_PATH = Path(r"D:\archive\data3a")

# Destination for cleaned dataset
DESTINATION_PATH = Path(r"D:\car-accident-mlops\data\processed")

# Dataset classes
CLASSES = {
    "01-minor": "Minor",
    "02-moderate": "Moderate",
    "03-severe": "Severe"
}

SPLITS = ["training", "validation"]


def calculate_hash(image_path):
    """Calculate SHA-256 hash of an image file."""

    sha256 = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def clean_dataset():

    print("\n========== DUPLICATE CLEANING ==========\n")

    # Create destination folders
    DESTINATION_PATH.mkdir(parents=True, exist_ok=True)

    seen_hashes = {}
    duplicate_records = []

    total_images = 0
    copied_images = 0
    duplicate_count = 0

    for split in SPLITS:

        print(f"\nProcessing {split}...")
        
        for folder, label in CLASSES.items():

            source_folder = SOURCE_PATH / split / folder
            destination_folder = DESTINATION_PATH / split / folder

            destination_folder.mkdir(parents=True, exist_ok=True)

            for image_path in source_folder.iterdir():

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in [".jpg", ".jpeg"]:
                    continue

                total_images += 1

                file_hash = calculate_hash(image_path)

                if file_hash in seen_hashes:

                    duplicate_count += 1

                    original = seen_hashes[file_hash]

                    duplicate_records.append({
                        "duplicate": str(image_path),
                        "original": str(original),
                        "split": split,
                        "class": label
                    })

                    print(f"Duplicate skipped: {image_path.name}")

                    continue

                # Remember this unique image
                seen_hashes[file_hash] = image_path

                # Copy without modifying original
                shutil.copy2(
                    image_path,
                    destination_folder / image_path.name
                )

                copied_images += 1

    # Save duplicate report
    report_path = DESTINATION_PATH / "duplicate_report.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        fieldnames = [
            "duplicate",
            "original",
            "split",
            "class"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(duplicate_records)

    # Print summary
    print("\n========== CLEANING SUMMARY ==========")

    print(f"Total images checked: {total_images}")
    print(f"Unique images copied: {copied_images}")
    print(f"Duplicate images skipped: {duplicate_count}")

    print(f"\nCleaned dataset location:")
    print(DESTINATION_PATH)

    print(f"\nDuplicate report:")
    print(report_path)


if __name__ == "__main__":
    clean_dataset()
