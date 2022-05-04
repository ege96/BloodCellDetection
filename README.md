# Inquiry Project
Using YOLOv5 to Detect and Classify Blood Cells

Interactive website - https://share.streamlit.io/pogman96/bloodcelldetectionstreamlit/main/main.py

# Dataset
https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset

## Requirements
### YOLOv5 - https://github.com/ultralytics/yolov5
### Python = 3.8.13
### Anaconda
### Cuda capable GPU or CPU

## Setup
Initial File Directory
```
.
├── imageProcessing.py
├── main.py
└── data/
    ├── images/
    │   └── PLACE DATASET HERE
    └── annotations.csv
```

```
conda env create -f environment.yml
conda activate pytorch
```

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

```bash
python imageProcessing.py
```

```bash
cd yolov5
bash data/scripts/download_weights.sh
```

## Training
```bash
cd yolov5
python train.py --data data/cellDetection.yaml --batch-size 4 --epochs 300 --img-size 640 --project runs/train --name cellDetection --weights yolov5x.pt --device 0
```
## Testing
```bash
cd yolov5
python detect.py --weights runs/train/cellDetection/weights/best.pt --source ../data/images/IMAGENAMEHERE.png --name cellDetection --project runs/detect
```
or
(In parent directory)
```bash
python main.py
```
## Pre-trained weights
On releases page

Place in yolov5 folder
