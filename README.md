# Trash/Waste Image Segmentation with YOLOv8

## 1. Project Title

Trash/Waste Image Segmentation with Fine-Tuned Ultralytics YOLOv8

## 2. Problem Definition

This project studies how to fine-tune a pretrained neural network for image
segmentation. The target task is trash or waste segmentation, where the model
must identify waste objects in an image and predict their object boundaries.

This is more detailed than simple classification because the output is not only
the object category, but also the object location and shape.

## 3. Motivation

Trash and waste detection is a meaningful computer vision problem because it can
support applications such as:

- smart waste collection
- environmental monitoring
- recycling support systems
- automatic scene understanding in public spaces

From an academic perspective, this project is a practical way to study transfer
learning, segmentation, dataset preparation, model evaluation, and experimental
workflow design.

## 4. Selected Architecture

The selected architecture is Ultralytics YOLO segmentation, with
`yolov8n-seg.pt` as the default starting model.

This model was selected because it:

- is pretrained and ready for transfer learning
- supports segmentation in addition to detection
- is lightweight compared to larger models
- is practical for student experiments on limited hardware
- has a clean Python interface for training, validation, and prediction

## 5. What is Fine-Tuning?

Fine-tuning is the process of taking a model that has already been pretrained on
a large dataset and continuing training it on a new task-specific dataset.

In this project, fine-tuning means:

- starting from pretrained YOLO segmentation weights
- replacing general knowledge with task-specific knowledge about trash images
- adapting the model to the custom dataset without training from scratch

This approach is useful because it usually needs less data, less training time,
and fewer computational resources than full training from the beginning.

## 6. YOLO Segmentation Architecture Overview

YOLO segmentation extends an object detection pipeline by adding segmentation
outputs for each detected object.

### Backbone

The backbone is the feature extraction part of the network. It reads the input
image and learns visual patterns such as edges, textures, shapes, and object
parts.

### Neck

The neck combines features from different image scales. This helps the model
detect both small and large objects more effectively by passing multi-scale
information to later stages.

### Head

The head produces the final prediction outputs. In a YOLO-style model, the head
predicts object-related information such as class confidence and bounding box
information.

### Segmentation Output

For segmentation, the model also predicts object masks. These masks represent
the shape of each detected object at the pixel or polygon level, which makes the
output more informative than a bounding box alone.

## 7. Dataset Format

This project expects a YOLO segmentation dataset with the following structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Important notes:

- each image should have a matching label file with the same filename stem
- the `data.yaml` file defines the split paths and class names
- segmentation labels store polygon point coordinates for each object
- the dataset is not included in this repository

An example configuration file is provided in `config/data.yaml.example`.

## 8. Training Pipeline

The planned pipeline for the project is:

1. Prepare or export the dataset in YOLO segmentation format.
2. Check the dataset structure with `scripts/check_dataset.py`.
3. Test the Python environment with `test_installation.py`.
4. Fine-tune the pretrained `yolov8n-seg.pt` model with `train.py`.
5. Run inference with `predict.py`.
6. Evaluate the model with `evaluate.py`.
7. Summarize observations, limitations, and future improvements.

## 9. Evaluation Metrics

The final evaluation metrics will be added after training and testing are
completed.

Planned metrics to report:

- validation loss: `[to be added]`
- segmentation mAP: `[to be added]`
- precision: `[to be added]`
- recall: `[to be added]`
- qualitative prediction examples: `[to be added]`

No real performance claims are included yet because the experiments are still in
progress.

## 10. Challenges Encountered

Some expected or common challenges in this type of project are:

- preparing a clean segmentation dataset
- making sure every image has a correct label file
- handling class imbalance or limited training data
- tuning epochs, batch size, and image size for available hardware
- avoiding overfitting on a small custom dataset
- interpreting segmentation errors in difficult scenes

This section can be updated later with project-specific challenges from the
actual experiments.

## 11. What I Learned

This project is designed to help build understanding in the following areas:

- how transfer learning works in practice
- how YOLO segmentation differs from standard classification
- how dataset structure affects training success
- how to organize a machine learning experiment clearly
- how to evaluate a segmentation system responsibly

This section should be updated at the end of the project with final reflections.

## 12. How to Run the Project

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Test the Installation

```bash
python test_installation.py
```

### Check the Dataset

```bash
python scripts/check_dataset.py
```

### Train the Model

```bash
python train.py
```

Example:

```bash
python train.py --epochs 10 --batch 4
```

### Run Prediction

```bash
python predict.py
```

Example:

```bash
python predict.py --source sample_images/test.jpg --conf 0.4
```

### Run Evaluation

```bash
python evaluate.py
```

## 13. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── evaluate.py
├── test_installation.py
├── config/
│   └── data.yaml.example
├── docs/
│   ├── presentation_outline.md
│   └── project_notes.md
├── results/
├── sample_images/
└── scripts/
    └── check_dataset.py
```

## 14. Future Improvements

Possible next steps for the project include:

- collecting a larger and more diverse waste dataset
- comparing `yolov8n-seg` with larger YOLO segmentation variants
- improving annotation quality and class definitions
- testing data augmentation strategies
- analyzing failure cases in more detail
- deploying the model in a small demo application

## Notes on Repository Contents

- The dataset is not included in this repository because datasets can be large.
- Trained model weights are not included in this repository because weight files
  can also be large.
- The repository is intended to contain code, configuration templates, and
  documentation for the academic project workflow.
