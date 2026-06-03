from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "yolov8n-seg.pt"
DEFAULT_DATA = "dataset/data.yaml"
DEFAULT_EPOCHS = 50
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 8
DEFAULT_PROJECT = "results"
DEFAULT_RUN_NAME = "trash_segmentation_yolov8n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained YOLO segmentation model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Pretrained model or local weights path.",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to the YOLO dataset configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--img-size",
        "--imgsz",
        type=int,
        dest="imgsz",
        default=DEFAULT_IMAGE_SIZE,
        help="Input image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int | None:
    if args.epochs <= 0:
        print("Error: '--epochs' must be a positive integer.")
        return 1

    if args.imgsz <= 0:
        print("Error: '--img-size' must be a positive integer.")
        return 1

    if args.batch <= 0:
        print("Error: '--batch' must be a positive integer.")
        return 1

    return None


def print_configuration(args: argparse.Namespace) -> None:
    print("Training configuration")
    print("----------------------")
    print(f"Model:      {args.model}")
    print(f"Data file:  {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Project:    {DEFAULT_PROJECT}")
    print(f"Run name:   {DEFAULT_RUN_NAME}")
    print()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    validation_error = validate_args(args)

    if validation_error is not None:
        return validation_error

    try:
        from ultralytics import YOLO
    except Exception as error:
        print("Error: ultralytics is not available in this environment.")
        print(f"Reason: {error}")
        print("Install the project dependencies with 'pip install -r requirements.txt'.")
        return 1

    if not data_path.is_file():
        print(f"Error: dataset configuration file was not found at '{data_path}'.")
        print("Create your dataset locally and add a valid YOLO data.yaml file first.")
        print("You can use 'config/data.yaml.example' as a starting template.")
        return 1

    print_configuration(args)

    try:
        model = YOLO(args.model)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=DEFAULT_PROJECT,
            name=DEFAULT_RUN_NAME,
        )
    except Exception as error:
        print("Training could not start.")
        print(f"Reason: {error}")
        return 1

    print("Training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
