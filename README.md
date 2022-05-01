# Inquiry Project
Using YOLOv5 to Detect and Classify Blood Cells and Platelets

# Dataset
https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset

## Requirements
### YOLOv5 - https://github.com/ultralytics/yolov5
### Python = 3.8.13
### Anaconda
### Cuda capable GPU or CPU

## Setup
```
conda env create -f environment.yml
conda activate pytorch
```

```bash
python imageProcessing.py
```

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

```bash
cd yolov5
bash data/scripts/download_weights.sh
```
## Training
```bash
python train.py --data data/cellDetection.yaml --batch-size -1 --epochs 300 --img-size 640 --project runs/train --name cellDetection --weights yolov5x.pt --device 0
```
## Testing
```bash
python detect.py --weights runs/train/cellDetection/weights/best.pt --source ../data/images/IMAGENAMEHERE.png --name cellDetection --project runs/detect
```
or
(In parent directory)
```bash
python main.py
```
