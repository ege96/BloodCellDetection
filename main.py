import torch
import os
import cv2
import random

#output image dimensions
RESIZE_W = 1000
RESIZE_H = 1000

# function to generate output image
def draw(image:str, labelVals:list, thresh:float):
    img = cv2.imread(image)
    img = cv2.resize(img, (RESIZE_W, RESIZE_H))


    for i in labelVals:
        # checking confidence threshold
        if i[-1] < thresh:
            continue
        cell = i[0]
        xc = float(i[1]) * RESIZE_W
        yc = float(i[2]) * RESIZE_H
        w = float(i[3]) * RESIZE_W
        h = float(i[4]) * RESIZE_H
    
        img = cv2.rectangle(img, (int(xc - w/2), int(yc - h/2)), (int(xc + w/2), int(yc + h/2)), (0,0,255), 1)
        img = cv2.putText(img, f"{cell} {round(i[-1], 2)}", (int(xc - w/2), int(yc - h/2 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 1)
        
    return img

# loading detection model
model = torch.hub.load("yolov5", "custom", path="yolov5/runs/train/cellDetection/weights/best.pt", source="local")

imgs = os.listdir("data/images")  #allImages
# sorting for aesthetics
imgs = sorted(imgs, key=lambda x: int(x.split("-")[-1].split(".")[0]))


while True:
    usr = input("Classify an image\n1. Enter a specific name\n2. Random select an image\n3. Display all images\n>>> ")
    if usr == "":
        break
    
    if usr.isnumeric():
        usr = int(usr)
        if usr in range(1,4):
            if usr == 1:
                name = input("\nFile name (with file extension): ")

            elif usr == 2:
                name = random.choice(imgs)

            elif usr == 3:
                for i in imgs:
                    print(i)
                print()
                continue
            
            # getting path to image file
            path = os.path.join("data/images", name)

            # getting image dimensions
            imgDimensions = cv2.imread(path).shape
            W = imgDimensions[0]
            H = imgDimensions[1]

            # applying trained model to image
            res = model(path)

            # results processing
            values = []
            for i in res.pandas().xyxy[0].values:
                temp = [i[-1], ((i[0] + i[2])/2) / W, ((i[1] + i[3])/2) / H, (i[2]-i[0]) / W, (i[3] - i[1]) / H, i[-3]]
                values.append(temp)

            # display image
            image = draw(path, values, 0.7)
            cv2.imshow("result", image)
            cv2.waitKey(0)

        else:
            continue

    else:
        continue
    