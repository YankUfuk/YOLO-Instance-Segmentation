import sys


def main() -> int:
    try:
        import ultralytics
        from ultralytics import YOLO
    except Exception as error:
        print("Ultralytics installation test failed.")
        print("Could not import 'ultralytics' or 'YOLO' from 'ultralytics'.")
        print(f"Reason: {error}")
        print("Install the dependencies with 'pip install -r requirements.txt'.")
        return 1

    print(f"Installed ultralytics version: {ultralytics.__version__}")

    try:
        model = YOLO("yolov8n-seg.pt")
        if model is None:
            raise RuntimeError("The model object was not created.")
    except Exception as error:
        print("Ultralytics installation test failed.")
        print("The pretrained model 'yolov8n-seg.pt' could not be loaded.")
        print(f"Reason: {error}")
        print("Check your internet connection, package installation, and model path.")
        return 1

    print("Success: ultralytics is installed and 'yolov8n-seg.pt' loaded correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
