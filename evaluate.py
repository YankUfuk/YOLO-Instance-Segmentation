from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO segmentation model on a dataset split."
    )
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="Model weights path or pretrained model name.",
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Path to the YOLO dataset configuration file.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)

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
        return 1

    print("Evaluation configuration")
    print("------------------------")
    print(f"Model:      {args.model}")
    print(f"Data file:  {args.data}")
    print(f"Split:      {args.split}")
    print(f"Image size: {args.imgsz}")
    print("Project:    results")
    print("Run name:   evaluation")
    print()

    try:
        model = YOLO(args.model)
        metrics = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            project="results",
            name="evaluation",
        )
    except Exception as error:
        print("Evaluation failed.")
        print(f"Reason: {error}")
        return 1

    print("Evaluation completed successfully.")
    print(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
