import pandas as pd
import os
import random
import cv2

# directory locations
trainDir = "training/"
testDir = "testing/"
allImagesDir = "images/"
dirs = ["images", "labels"]
dataDir = "data/"
yaml_file = "yolov5/data/cellDetection.yaml"

os.chdir(dataDir)

def copyImages(baseDir, finalDir, imgNames):
    for i in imgNames:
        originalFile = os.path.join(baseDir, i)
        outputFile = os.path.join(finalDir, "images", i)
        original = cv2.imread(originalFile)
        reSized = cv2.resize(original, SIZE)
        cv2.imwrite(outputFile, reSized)


def writeLabels(imageList, directory):
    for i in imageList:
        file = os.path.join(directory, "labels", i.split(".")[0] + ".txt")
        with open(file, "w") as f:
            for j in anno[anno["image"] == i].values:
                f.write(
                    f"{cellDict[j[5]]} {((j[3]+j[1])/2) / W} {((j[4]+j[2])/2) / H} {(j[3]-j[1]) / W} {(j[4]-j[2]) / W} \n")

# dictionary for cell id's
cellDict = {"rbc": 0, "wbc": 1}
cells = list(cellDict.keys())

allImgs = os.listdir("images")

random.shuffle(allImgs)

# spliting images into 2 parts
training = allImgs[:80]
testing = allImgs[80:]

# reading annotations for labeling
anno = pd.read_csv("annotations.csv")

# making folders
for i in [trainDir, testDir]:
    for j in dirs:
        try:
            os.makedirs(os.path.join(i, j))
        except FileExistsError:
            print("Folder already exists")

# getting size of dataset iamges
temp = cv2.imread("images/"+allImgs[0]).shape
W = temp[0]
H = temp[1]

writeLabels(training, trainDir)
writeLabels(testing, testDir)

# resizing images for YOLOv5
RESIZE_W = 640
RESIZE_H = 640
SIZE = (RESIZE_W, RESIZE_H)

# copying images to training and testing directory
copyImages(allImagesDir, testDir, testing)
copyImages(allImagesDir, trainDir, training)

os.chdir("..")
trainingImagesDir = os.path.join("../data", trainDir, "images")
testingImagesDir = os.path.join("../data", testDir, "images")

names = ", ".join([f"\"{i}\"" for i in cells])

# setting up configuration file for training
with open(yaml_file, "w") as f:
    f.write(f"train: {trainingImagesDir}\n")
    f.write(f"val: {testingImagesDir}\n")
    f.write(f"nc: {len(cells)}\n")
    f.write(f"names: [{names}]\n")
