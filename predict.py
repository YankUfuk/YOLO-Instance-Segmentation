from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "results/trash_segmentation_yolov8n/weights/best.pt"
DEFAULT_SOURCE = "dataset/test/images"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 640
DEFAULT_PROJECT = "results"
DEFAULT_RUN_NAME = "test_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run segmentation prediction with a YOLO model."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the trained YOLO model weights.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Image file or folder to use for prediction.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--img-size",
        "--imgsz",
        type=int,
        dest="imgsz",
        default=DEFAULT_IMAGE_SIZE,
        help="Input image size for inference.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int | None:
    if args.conf < 0 or args.conf > 1:
        print("Error: '--conf' must be between 0.0 and 1.0.")
        return 1

    if args.imgsz <= 0:
        print("Error: '--img-size' must be a positive integer.")
        return 1

    return None


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)
    validation_error = validate_args(args)

    if validation_error is not None:
        return validation_error

    if not model_path.is_file():
        print(f"Error: model file was not found at '{model_path}'.")
        print("Train the model first or pass a valid weights file with --model.")
        return 1

    if not source_path.exists():
        print(f"Error: prediction source was not found at '{source_path}'.")
        print("Provide a valid image file or folder with --source.")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as error:
        print("Error: ultralytics is not available in this environment.")
        print(f"Reason: {error}")
        print("Install the project dependencies with 'pip install -r requirements.txt'.")
        return 1

    print("Prediction configuration")
    print("------------------------")
    print(f"Model:      {args.model}")
    print(f"Source:     {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"Image size: {args.imgsz}")
    print(f"Project:    {DEFAULT_PROJECT}")
    print(f"Run name:   {DEFAULT_RUN_NAME}")
    print()

    try:
        model = YOLO(str(model_path))
        model.predict(
            source=str(source_path),
            conf=args.conf,
            imgsz=args.imgsz,
            project=DEFAULT_PROJECT,
            name=DEFAULT_RUN_NAME,
            save=True,
            exist_ok=True,
        )
    except Exception as error:
        print("Prediction failed.")
        print(f"Reason: {error}")
        return 1

    print("Prediction completed successfully.")
    print(f"Saved outputs to: {Path(DEFAULT_PROJECT) / DEFAULT_RUN_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
