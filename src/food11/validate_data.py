import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from data import CLASSES, SPLITS


def validate_dataset(root: Path, min_per_class: int, tau: float) -> bool:
    expected = set(CLASSES.values())
    counts = {split: Counter() for split in SPLITS}
    hashes = {}
    errors = {
        "folders": [],
        "images": [],
        "counts": [],
        "duplicates": [],
        "distribution": [],
    }

    for split in SPLITS:
        split_dir = root / split

        actual = (
            {p.name for p in split_dir.iterdir() if p.is_dir()}
            if split_dir.is_dir()
            else set()
        )

        if actual != expected:
            errors["folders"].append(
                f"{split}: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

        for category in sorted(expected):
            folder = split_dir / category

            if not folder.is_dir():
                continue

            images = sorted(p for p in folder.iterdir() if p.is_file())
            counts[split][category] = len(images)

            if len(images) < min_per_class:
                errors["counts"].append(
                    f"{split}/{category}: {len(images)} images"
                )

            for path in images:
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB" or img.size != (128, 128):
                            errors["images"].append(
                                f"{path}: {img.mode}, {img.size}"
                            )
                except (OSError, ValueError) as exc:
                    errors["images"].append(f"{path}: {exc}")

                digest = hashlib.md5(path.read_bytes()).hexdigest()
                hashes.setdefault(digest, {}).setdefault(split, []).append(
                    str(path)
                )

    for locations in hashes.values():
        if len(locations) > 1:
            errors["duplicates"].append(
                ", ".join(
                    paths[0] for paths in locations.values()
                )
            )

    train_total = sum(counts["training"].values())
    val_total = sum(counts["validation"].values())

    if not train_total or not val_total:
        errors["distribution"].append("Empty training or validation split")
    else:
        for category in sorted(expected):
            train_share = counts["training"][category] / train_total
            val_share = counts["validation"][category] / val_total

            if abs(train_share - val_share) > tau:
                errors["distribution"].append(
                    f"{category}: training={train_share:.3f}, "
                    f"validation={val_share:.3f}"
                )

    labels = {
        "folders": "11 class folders per split",
        "images": "RGB images, 128x128",
        "counts": f"At least {min_per_class} images per class",
        "duplicates": "No duplicates across splits",
        "distribution": f"Class distribution difference <= {tau}",
    }

    success = True

    for key, label in labels.items():
        problems = errors[key]

        if problems:
            success = False
            print(f"FAIL - {label}: {len(problems)} issue(s)")
            for problem in problems[:5]:
                print(f"  {problem}")
        else:
            print(f"PASS - {label}")

    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"Dataset not found: {args.dataset}")
        return 1

    return 0 if validate_dataset(
        args.dataset, args.min_per_class, args.tau
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
