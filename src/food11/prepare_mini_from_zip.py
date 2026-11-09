from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "data" / "food11.zip"
OUTPUT = ROOT / "data" / "food11_processed_mini"

CLASSES = {
    "0": "Bread",
    "1": "Dairy product",
    "2": "Dessert",
    "3": "Egg",
    "4": "Fried food",
    "5": "Meat",
    "6": "Noodles-Pasta",
    "7": "Rice",
    "8": "Seafood",
    "9": "Soup",
    "10": "Vegetable-Fruit",
}

SPLITS = ["training", "evaluation", "validation"]

with ZipFile(ZIP_PATH) as archive:
    filenames = sorted(archive.namelist())

    for split in SPLITS:
        counts = Counter()
        prefix = f"food11 dataset/{split}/"

        for filename in filenames:
            if not filename.startswith(prefix):
                continue

            name = Path(filename).name
            if not name.lower().endswith(".jpg"):
                continue

            class_id = name.split("_")[0]
            if class_id not in CLASSES:
                continue

            category = CLASSES[class_id]
            if counts[category] >= 100:
                continue

            destination = OUTPUT / split / category / name
            destination.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(BytesIO(archive.read(filename))) as image:
                image.convert("RGB").resize((128, 128)).save(destination)

            counts[category] += 1

        print(f"{split}: {sum(counts.values())} images")

print("Mini dataset created successfully!")
