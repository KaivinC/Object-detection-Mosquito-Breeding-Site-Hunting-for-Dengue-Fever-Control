import pandas as pd
from tqdm import tqdm
from glob import glob
import cv2
import numpy as np
import os
from math import ceil
from utils1.utils import STANDARD_COLORS, standard_to_bgr,plot_one_box

results = {}
results["box_x"] = []
results["box_y"] = []
results["box_w"] = []
results["box_h"] = []
results["score"] = []
results["label"] = []
results["name"] = [] 

names=[ "aquarium","bottle","bowl","box","bucket","plastic_bag","plate","styrofoam","tire","toilet","tub","washing_machine","water_tower"]

pred_path = "/home/wjhuang/yolov5/yolov5/runs/test/exp44/labels/*"
data_path = "/home/wjhuang/yolov5/test_cdc/test_images"
color_list = standard_to_bgr(STANDARD_COLORS)
idx = 0
for file_path in tqdm(glob(pred_path)):
    img = cv2.imread(os.path.join(data_path,file_path.split("/")[-1].split(".")[0]+".jpg"))
    img_h, img_w, img_c = img.shape
    fp = open(file_path,"r")
    for line in iter(fp):
        nms_class = int(line[:-1].split(" ")[0])
        x_center = float(line[:-1].split(" ")[1])
        y_center = float(line[:-1].split(" ")[2])
        ww = float(line[:-1].split(" ")[3])
        hh = float(line[:-1].split(" ")[4])
        x = ceil((x_center-ww/2)*img_w)
        y = ceil((y_center-hh/2)*img_h)
        w = ceil(ww*img_w)
        h = ceil(hh*img_h)
        nms_scores = float(line[:-1].split(" ")[5])
        results["box_x"].append(x)
        results["box_y"].append(y)
        results["box_w"].append(w)
        results["box_h"].append(h)
        results["score"].append(nms_scores)
        results["label"].append(nms_class+1)
        results["name"].append(file_path.split("/")[-1].split(".")[0]+".jpg")

        label = names[nms_class]
        #plot_one_box(img, [x, y, x+w, y+h], label=label,score=nms_scores,color=color_list[nms_class])
    
    #cv2.imwrite(f'test/{file_path.split("/")[-1].split(".")[0]}.jpg', img)
    idx += 1
    fp.close()

result_dict = {"image_filename": results["name"],  
    "label_id": results["label"],
    "x":results["box_x"],
    "y":results["box_y"],
    "w":results["box_w"],
    "h":results["box_h"],
    "confidence":results["score"]
   }
select_df = pd.DataFrame(result_dict)
print(select_df)
select_df.to_csv("submission.csv", index=False)

    