# Project Notes

## Goal

Fine-tune a pretrained YOLO segmentation model for trash or waste segmentation.

## Research Direction

- Compare pretrained segmentation transfer learning with training from scratch.
- Track segmentation quality with standard validation metrics.
- Record how dataset quality affects model performance.

## Suggested Experiments

- Train with different epoch values.
- Compare `yolov8n-seg` with a larger segmentation variant if hardware allows.
- Test different image sizes and batch sizes.

## YOLO Segmentation Dataset Structure

A YOLO segmentation dataset usually contains separate `train`, `valid`, and `test`
folders. Inside each split, there is an `images/` folder for image files and a
`labels/` folder for annotation text files.

For segmentation, each label file matches one image filename and stores the class
index followed by polygon points that describe the object mask. The `data.yaml`
file tells YOLO where the dataset splits are located and which class names are used.

## Notes to Fill In During the Project

- Dataset source and collection method
- Annotation workflow
- Hardware used for training
- Best validation results
- Common failure cases
- Final conclusions
