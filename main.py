import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt


def writeLabels(imageList, directory):
    for i in imageList:
        file = os.path.join(directory, "labels", i.split(".")[0] + ".txt")
        with open(file, "w") as f:
            for j in anno[anno["image"] == i].values:
                f.write(
                    f"{cellDict[j[5]]} {((j[3]+j[1])/2) / W} {((j[4]+j[2])/2) / H} {(j[3]-j[1]) / W} {(j[4]-j[2]) / W} \n")


trainDir = "training/"
testDir = "testing/"
dirs = ["images", "labels"]
dataDir = "data/"
os.chdir(dataDir)

cellDict = {"rbc": 0, "wbc": 1}
cells = list(cellDict.keys())

allImgs = []
for i in os.listdir("images"):
    allImgs.append(i)

training = allImgs[:80]
testing = allImgs[80:]

anno = pd.read_csv("annotations.csv")

for i in [trainDir, testDir]:
    for j in dirs:
        try:
            os.makedirs(os.path.join(i, j))
        except FileExistsError:
            print("Folder already exists")

temp = cv2.imread("images/"+allImgs[0]).shape
W = temp[0]
H = temp[1]

writeLabels(training, trainDir)
writeLabels(testing, testDir)

RESIZE_W = 640
RESIZE_H = 640
