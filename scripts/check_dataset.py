from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_SUFFIX = ".txt"
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO segmentation dataset folder."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="dataset",
        help="Path to the dataset folder. Default: dataset",
    )
    return parser.parse_args()


def count_files(folder: Path, suffixes: set[str] | None = None) -> list[Path]:
    if not folder.is_dir():
        return []

    files = [path for path in folder.iterdir() if path.is_file()]
    if suffixes is None:
        return files
    return [path for path in files if path.suffix.lower() in suffixes]


def split_summary(dataset_dir: Path, split: str) -> dict[str, Any]:
    split_dir = dataset_dir / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    image_files = count_files(images_dir, IMAGE_SUFFIXES)
    label_files = count_files(labels_dir, {LABEL_SUFFIX})

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    label_only = sorted(label_stems - image_stems)

    return {
        "split_dir_exists": split_dir.is_dir(),
        "images_dir_exists": images_dir.is_dir(),
        "labels_dir_exists": labels_dir.is_dir(),
        "image_count": len(image_files),
        "label_count": len(label_files),
        "label_only": label_only,
    }


def print_header(dataset_dir: Path) -> None:
    print("YOLO Segmentation Dataset Check")
    print("===============================")
    print(f"Dataset directory: {dataset_dir}")
    print(f"data.yaml found:   {'yes' if (dataset_dir / 'data.yaml').is_file() else 'no'}")
    print()


def print_summary_table(results: dict[str, dict[str, Any]]) -> None:
    headers = ("Split", "Split Dir", "Images Dir", "Labels Dir", "Images", "Labels")
    rows = []

    for split in SPLITS:
        summary = results[split]
        rows.append(
            (
                split,
                yes_no(summary["split_dir_exists"]),
                yes_no(summary["images_dir_exists"]),
                yes_no(summary["labels_dir_exists"]),
                str(summary["image_count"]),
                str(summary["label_count"]),
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_line = " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator_line = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print()


def yes_no(value: object) -> str:
    return "yes" if value else "no"


def print_warnings(dataset_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    warnings_found = False

    if not (dataset_dir / "data.yaml").is_file():
        warnings_found = True
        print("Warning: 'data.yaml' is missing from the dataset root.")

    for split in SPLITS:
        summary = results[split]
        image_count = int(summary["image_count"])
        label_count = int(summary["label_count"])
        label_only = list(summary["label_only"])

        if image_count > 0 and label_count == 0:
            warnings_found = True
            print(f"Warning: '{split}' has images but no label files.")

        if label_only:
            warnings_found = True
            print(
                f"Warning: '{split}' has {len(label_only)} label file(s) without matching image names."
            )
            preview = ", ".join(label_only[:5])
            print(f"Examples: {preview}")

        if not summary["split_dir_exists"]:
            warnings_found = True
            print(f"Warning: '{split}' split folder is missing.")
        elif not summary["images_dir_exists"] or not summary["labels_dir_exists"]:
            warnings_found = True
            print(f"Warning: '{split}' is missing an 'images' or 'labels' folder.")

    if not warnings_found:
        print("No structural problems were found in the dataset folders.")


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    results = {split: split_summary(dataset_dir, split) for split in SPLITS}

    print_header(dataset_dir)
    print_summary_table(results)
    print_warnings(dataset_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
