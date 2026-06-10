# OOS-Detect: Empty Shelf Area Detection Using YOLOv8n

This project implements an object detection model to identify empty shelf areas in retail shelves as an early indicator of out-of-stock (OOS) conditions. The model uses YOLOv8n and is trained on a publicly available Roboflow dataset containing annotated supermarket shelf images.

## Project Overview

Out-of-stock (OOS) occurs when a product is not available on the shelf where customers expect to find it. This issue can lead to missed sales opportunities, lower customer satisfaction, and delays in replenishment. Through computer vision, empty shelf areas can be detected automatically from retail shelf images.

This project focuses on detecting fully empty shelf areas using bounding boxes. The output of the model includes the detected OOS area, class label, and confidence score.

## Dataset

The dataset used in this project is:

**Supermarket Empty Shelf Detector**
Source: Roboflow Universe
Link: https://universe.roboflow.com/fyp-ormnr/supermarket-empty-shelf-detector

Dataset details:

* Format: YOLOv8
* Total images: 497 shelf images
* Train set: 350 images
* Validation set: 97 images
* Test set: 50 images
* Number of classes: 1
* Class label: `100- O-O-S`
* Class meaning: fully empty out-of-stock shelf area

## Model

The model used in this project is **YOLOv8n**, a lightweight version of YOLOv8 designed for efficient object detection.

Model configuration:

* Model architecture: YOLOv8n
* Task: Object Detection
* Pretrained weights: COCO
* Image size: 640 × 640 px
* Epochs: 100
* Batch size: 16
* Output: bounding box, class label, and confidence score

## Methodology

The workflow of this project consists of the following steps:

1. Download the dataset from Roboflow in YOLOv8 format.
2. Inspect the dataset structure, labels, annotations, and train-validation-test split.
3. Fine-tune the YOLOv8n pretrained model on the training set.
4. Monitor model performance using the validation set.
5. Evaluate the final model on the test set.
6. Visualize detection results using bounding boxes.
7. Interpret the model performance using object detection metrics.

## Evaluation Results

The model was evaluated on the test set consisting of 50 images and 356 OOS instances.

| Metric    |  Value |
| --------- | -----: |
| Precision | 0.8658 |
| Recall    | 0.8933 |
| F1-score  | 0.8793 |
| mAP@50    | 0.9181 |
| mAP@50-95 | 0.6839 |

The results show that the model performs well in detecting empty shelf areas. The recall value indicates that most actual OOS areas were successfully detected, while the mAP@50 score shows strong localization performance at an IoU threshold of 0.5.

## How to Run

### 1. Install Dependencies

```bash
pip install ultralytics roboflow
```

### 2. Download Dataset from Roboflow

Use your Roboflow API key to download the dataset in YOLOv8 format.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("fyp-ormnr").project("supermarket-empty-shelf-detector")
version = project.version(3)
dataset = version.download("yolov8")
```

> Note: Do not expose your Roboflow API key in a public repository.

### 3. Train YOLOv8n Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="oos_yolov8n"
)
```

### 4. Evaluate the Model

```python
model.val(
    data=f"{dataset.location}/data.yaml",
    split="test"
)
```

### 5. Run Prediction

```python
model.predict(
    source=f"{dataset.location}/test/images",
    conf=0.25,
    save=True,
    name="oos_test_predict"
)
```

## Sample Output

The model predicts empty shelf areas using bounding boxes. Each predicted bounding box contains:

* class label: `100- O-O-S`
* confidence score
* location of the detected empty shelf area

## Key Findings

* YOLOv8n achieved strong detection performance on the test set.
* The model obtained an mAP@50 of 0.9181 and an F1-score of 0.8793.
* The approach shows potential as an early-stage visual shelf monitoring system for detecting out-of-stock areas.
* Since the dataset only contains one class, the model focuses specifically on detecting fully empty shelf areas.

## Limitations

This project has several limitations:

* The dataset is publicly sourced and may not fully represent Indonesian retail environments.
* The model only detects fully empty shelf areas and does not classify low-stock or nearly empty shelf conditions.
* The system is not yet integrated with real inventory, POS, or restocking notification systems.
* Model performance may vary depending on lighting conditions, camera angle, shelf type, and product arrangement.

## References

1. Jha, D., Mahjoubfar, A., & Joshi, A. (2022). *Designing an Efficient End-to-end Machine Learning Pipeline for Real-time Empty-shelf Detection*. arXiv:2205.13060.

2. Šikić, F., Kalafatić, Z., Subašić, M., & Lončarić, S. (2024). *Enhanced Out-of-Stock Detection in Retail Shelf Images Based on Deep Learning*. Sensors, 24(2), 693.

3. FYP. (2023). *Supermarket Empty Shelf Detector Dataset*. Roboflow Universe. https://universe.roboflow.com/fyp-ormnr/supermarket-empty-shelf-detector

## Author

**Naura Alya Salsabila**  
Statistics Undergraduate Student  
Universitas Indonesia
