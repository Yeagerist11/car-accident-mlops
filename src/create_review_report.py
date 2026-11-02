import csv
from pathlib import Path

output_path = Path("reports/human_review.csv")

review_data = [
    {
        "image": "download (18).jpeg",
        "original_label": "Moderate",
        "review_status": "Needs Review",
        "suggested_label": "Moderate / Minor",
        "notes": "Localized hood dents; close-up makes assessment difficult."
    },
    {
        "image": "0023.JPEG",
        "original_label": "Moderate",
        "review_status": "Keep",
        "suggested_label": "Moderate",
        "notes": "Bent bumper and crooked plate; underlying damage unclear."
    },
    {
        "image": "0028.JPEG",
        "original_label": "Moderate",
        "review_status": "Needs Review",
        "suggested_label": "Moderate / Severe",
        "notes": "Crushed rear bumper, partly detached."
    },
    {
        "image": "0050.JPEG",
        "original_label": "Moderate",
        "review_status": "Needs Review",
        "suggested_label": "Moderate / Severe",
        "notes": "Torn front bumper and headlight damage."
    },
    {
        "image": "download (21).jpeg",
        "original_label": "Moderate",
        "review_status": "Needs Review",
        "suggested_label": "Severe",
        "notes": "Extensive front-end damage; clearest Severe candidate."
    }
]

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=review_data[0].keys())
    writer.writeheader()
    writer.writerows(review_data)

print("Human review report created!")
print(f"Saved at: {output_path}")