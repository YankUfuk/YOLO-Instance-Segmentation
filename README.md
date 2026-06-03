# Small Trash Image Segmentation with YOLOv8

I wanted to understand how a pretrained deep learning model can be adapted to a
specific computer vision problem. Instead of training a model from scratch, I
used fine-tuning. The problem I chose was small trash segmentation: teaching a
model to find small trash objects in images and outline their shapes.

This repository is the practical result of that learning process. It is written
to be simple enough for a beginner to follow, while still being useful as a
GitHub project for a course assignment or presentation.

## Why This Project?

When people first hear about computer vision, they often think only about
"recognizing" what is inside an image. But there are actually different levels
of understanding:

- **Classification** answers: "What is in this image?"
- **Object detection** answers: "What objects are in this image, and where are they?"
- **Instance segmentation** answers: "What objects are in this image, where are they, and what exact shape does each object have?"

This project focuses on **instance segmentation** of small trash objects.

That matters because trash is often small, irregular, and visually messy. A
bounding box alone may not describe it very well. Segmentation is more useful
because it helps the model learn not only the object location, but also the
object boundary.

## The Model: YOLOv8n-seg

This project uses **Ultralytics YOLO** with the pretrained model
`yolov8n-seg.pt`.

It is important to understand that **YOLO is not just a program**. It is a
computer vision model architecture designed for tasks like detection and
segmentation.

The chosen architecture here is **YOLOv8n-seg**:

- `YOLOv8` is the model family
- `n` means **nano**, the lightweight version
- `seg` means it is the **segmentation** version

YOLOv8n-seg predicts:

- the class label
- the bounding box
- the confidence score
- the segmentation mask

I selected the nano version because it is smaller, faster, and more practical
for a student project running locally on a CPU.

## What Fine-Tuning Means

A pretrained model has already learned useful visual patterns from large
datasets. It has seen many examples of shapes, textures, edges, and object
structures.

Fine-tuning means we do not throw away that earlier knowledge. Instead, we
continue training the model on a smaller, task-specific dataset so it can adapt
to a new problem.

In this project, the pretrained YOLOv8n-seg model was adapted to a **single
class**:

- `small trash`

That makes fine-tuning a very practical approach. It is faster and usually more
realistic than trying to train a segmentation model from the beginning.

## The Dataset

The dataset used for this project comes from **Roboflow**.

It is a YOLO segmentation dataset with the following local structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

Local paths used in the project:

- `dataset/train/images`
- `dataset/train/labels`
- `dataset/valid/images`
- `dataset/valid/labels`
- `dataset/data.yaml`

Dataset size:

- train: `73` images and `73` labels
- valid: `23` images and `23` labels
- no separate test split

Because the dataset did not include a separate test folder, the **validation
set was also used for checking predictions and evaluation** during this initial
experiment.

The `dataset/` folder is ignored by Git, so anyone using this repository needs
to **download the dataset locally** and place it inside `dataset/`.

## How YOLO Uses the Dataset

One of the most important things I learned is that training is not just "giving
images to a model."

Each image has a matching label file:

- the **image** is the input
- the **label file** is the ground truth

During training, YOLO:

1. reads the image
2. predicts the class, bounding box, and segmentation mask
3. compares that prediction with the label file
4. calculates the error, called **loss**
5. updates the model weights

This update step is what makes learning happen.

For segmentation, the label files are especially important because they contain
**polygon or mask information**, not only boxes. That means the model is not
just learning where the object is, but also the object boundary.

## Initial Fine-Tuning Experiment

The goal of the first run was not to reach perfect accuracy. The goal was to
verify that the full fine-tuning pipeline works correctly from start to finish.

Training details:

- architecture: `YOLOv8n-seg`
- pretrained weights: `yolov8n-seg.pt`
- library: `Ultralytics YOLO`
- hardware: local CPU
- experiment type: initial pipeline verification

Training command used:

```bash
python train.py --epochs 3 --batch 2
```

This short run successfully completed and produced a trained weights file:

```text
best.pt
```

In other words, the fine-tuning workflow worked correctly: the model loaded,
the dataset was used, training finished, and the best checkpoint was saved.

## Observed Validation Metrics

These are the observed validation metrics from the 3-epoch run:

| Metric | Value |
| --- | ---: |
| Box Precision | 0.55 |
| Box Recall | 0.517 |
| Box mAP50 | 0.457 |
| Box mAP50-95 | 0.262 |
| Mask Precision | 0.525 |
| Mask Recall | 0.494 |
| Mask mAP50 | 0.446 |
| Mask mAP50-95 | 0.233 |

These numbers should be interpreted carefully.

Because the model was trained only for **3 epochs** on a **small dataset**, the
goal was not to achieve high final accuracy. The purpose was to demonstrate that
the fine-tuning process works and produces meaningful outputs.

## Repository Files

Here is what the main files do:

- `train.py`: loads `YOLOv8n-seg` and fine-tunes it on the dataset
- `scripts/check_dataset.py`: checks dataset structure, image counts, and label counts
- `predict.py`: runs the trained model on images and saves visual predictions
- `evaluate.py`: evaluates the model on a dataset split
- `requirements.txt`: lists the Python dependencies
- `README.md`: explains the project, workflow, and lessons learned

## How to Run the Project

### 1. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the YOLO segmentation dataset from Roboflow and place it under the
local `dataset/` folder.

### 4. Check the Dataset

```bash
python scripts/check_dataset.py
```

### 5. Train the Model

```bash
python train.py --epochs 3 --batch 2
```

### 6. Run Prediction

If you trained the model with this repository's default settings, the saved
weights are expected at:

```text
results/trash_segmentation_yolov8n/weights/best.pt
```

Example prediction command:

```bash
python predict.py --model results/trash_segmentation_yolov8n/weights/best.pt --source dataset/valid/images
```

If your environment saved weights under a different Ultralytics run directory,
update the `--model` path accordingly.

### 7. Optional Evaluation

Because this dataset has no separate test split, evaluation can be run on the
validation split:

```bash
python evaluate.py --model results/trash_segmentation_yolov8n/weights/best.pt --data dataset/data.yaml --split val
```

## What I Learned

This project helped me understand that deep learning is not just about calling a
model from a library.

Some of the main lessons were:

- training is a loop of prediction, loss calculation, backpropagation, and weight updates
- CNN-based models learn visual features such as edges, textures, shapes, and object boundaries
- fine-tuning makes it possible to reuse pretrained knowledge for a new task
- dataset format and label quality are as important as the model itself
- segmentation is more detailed than detection because the model learns object shape

## Main Challenge

The hardest part for me was understanding how YOLO uses image-label pairs during
training.

I had to realize that the label files are not just extra files sitting beside
the images. They are the **ground truth** that tells the model what the correct
answer should be.

For segmentation, this becomes even more interesting because the labels
represent object masks or polygons. That means the model is learning object
boundaries, not only rough object locations.

Understanding how YOLO compares predictions with labels and then updates the
weights was the most educational part of the project.

## Limitations

This is still a small and early experiment, so it has several limitations:

- small dataset
- only 3 epochs of training
- no separate test split
- CPU-only training
- results are initial experimental results, not production-level performance

## Future Improvements

If I continue this project, the next steps would be:

- train for more epochs
- use a larger dataset
- add a separate test split
- try stronger YOLO models such as `YOLOv8s-seg` or `YOLOv8m-seg`
- improve annotation quality
- compare results across different model sizes

## Repository Structure

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

## Notes

- The dataset is not included in the repository because datasets can be large.
- Trained weights are not included in the repository because model files can also be large.
- This project is a learning-focused fine-tuning experiment, not a production-ready system.
