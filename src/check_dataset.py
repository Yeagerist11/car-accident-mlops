
from pathlib import Path
from PIL import Image
import hashlib

# Location of our dataset
DATASET_PATH = Path(r"D:\archive\data3a")

# Dataset classes
CLASSES = {
    "01-minor": "Minor",
    "02-moderate": "Moderate",
    "03-severe": "Severe"
}

# Dataset splits
SPLITS = ["training", "validation"]


def check_dataset():

    print("\n========== DATASET QUALITY REPORT ==========\n")

    total_images = 0
    corrupted_images = 0

    image_sizes = {}

    # Store image hashes to identify exact duplicates
    image_hashes = {}
    duplicate_images = []

    for split in SPLITS:

        print(f"\n{split.upper()} DATASET")
        print("--------------------------------")

        for folder, label in CLASSES.items():

            class_path = DATASET_PATH / split / folder

            image_count = 0

            for image_path in class_path.iterdir():

                if image_path.suffix.lower() not in [".jpg", ".jpeg"]:
                    continue

                image_count += 1
                total_images += 1

                # Check for exact duplicate files
                file_hash = hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest()

                if file_hash in image_hashes:
                    duplicate_images.append(
                        (str(image_path), image_hashes[file_hash])
                    )
                else:
                    image_hashes[file_hash] = str(image_path)

                # Check image readability and dimensions
                try:
                    with Image.open(image_path) as image:

                        image_width, image_height = image.size

                        size = (image_width, image_height)

                        image_sizes[size] = (
                            image_sizes.get(size, 0) + 1
                        )

                        image.verify()

                except Exception:
                    corrupted_images += 1
                    print(f"Corrupted image: {image_path}")

            print(f"{label}: {image_count} images")

    # Final report
    print("\n========== SUMMARY ==========")

    print(f"Total images: {total_images}")
    print(f"Corrupted images: {corrupted_images}")
    print(f"Valid images: {total_images - corrupted_images}")

    print("\n========== IMAGE DIMENSIONS ==========")

    for size, count in sorted(
        image_sizes.items(),
        key=lambda item: item[1],
        reverse=True
    )[:10]:

        print(f"{size[0]} x {size[1]}: {count} images")

    print("\n========== DUPLICATE CHECK ==========")

    print(f"Number of duplicate files: {len(duplicate_images)}")

    for duplicate, original in duplicate_images[:10]:

        print(f"Duplicate: {duplicate}")
        print(f"Original:  {original}")


if __name__ == "__main__":
    check_dataset()
