from pathlib import Path
from shutil import rmtree
from PIL import Image

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "food11_raw"
PROCESSED_DIR = DATA_DIR / "food11_processed"
MINI_DIR = DATA_DIR / "food11_processed_mini"

IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

# Food-11 class mapping
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


def prepare_output_folders():
    # Remove old processed folders so rerunning the script starts clean
    for folder in [PROCESSED_DIR, MINI_DIR]:
        if folder.exists():
            rmtree(folder)

    # Create split/category structure
    for split in SPLITS:
        for category in CLASSES.values():
            (PROCESSED_DIR / split / category).mkdir(
                parents=True,
                exist_ok=True,
            )

            (MINI_DIR / split / category).mkdir(
                parents=True,
                exist_ok=True,
            )


def process_image(source_path, destination_path):
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image.save(destination_path)


def process_split(split):
    source_dir = RAW_DIR / split

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found: {source_dir}"
        )

    mini_counts = {category: 0 for category in CLASSES.values()}

    files = sorted(
        path for path in source_dir.iterdir()
        if path.is_file()
    )

    print(f"\nProcessing {split}: {len(files)} images")

    for index, image_path in enumerate(files, start=1):
        class_id = image_path.stem.split("_")[0]

        if class_id not in CLASSES:
            print(f"Skipping unknown file: {image_path.name}")
            continue

        category = CLASSES[class_id]

        processed_path = (
            PROCESSED_DIR
            / split
            / category
            / image_path.name
        )

        process_image(image_path, processed_path)

        # Copy/process at most 100 images per category into mini dataset
        if mini_counts[category] < MINI_LIMIT:
            mini_path = (
                MINI_DIR
                / split
                / category
                / image_path.name
            )

            process_image(image_path, mini_path)
            mini_counts[category] += 1

        if index % 1000 == 0:
            print(f"  Processed {index}/{len(files)}")


def main():
    print("Preparing Food-11 datasets...")
    print(f"Raw dataset: {RAW_DIR}")

    prepare_output_folders()

    for split in SPLITS:
        process_split(split)

    print("\nDone!")
    print(f"Processed dataset: {PROCESSED_DIR}")
    print(f"Mini dataset:      {MINI_DIR}")


if __name__ == "__main__":
    main()