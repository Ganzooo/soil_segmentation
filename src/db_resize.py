import numpy as np
import glob2
import math
import cv2
from pathlib import Path

data_path = '/dataset/Woodscape/soiling_dataset_nodist_150/'
dataset_type = 'train'

if data_path == "/dataset/Woodscape/soiling_dataset_nodist_150/":
    Hcrop = 736
    Wcrop = 992
else: 
    Hcrop = 224
    Wcrop = 320

if dataset_type == 'val':
    gtOutPath = data_path + 'test/gtLabels_crop/'
    rgbOutPath = data_path + 'test/rgbImages_crop/'
    gtRgbOutPath = data_path + 'test/rgbLabels_crop/'
else: 
    gtOutPath = data_path + 'train/gtLabels_crop/'
    rgbOutPath = data_path + 'train/rgbImages_crop/'
    gtRgbOutPath = data_path + 'train/rgbLabels_crop/'

if __name__ == "__main__":
    if dataset_type == 'train':
        in_feature_paths = list(sorted(Path(data_path).glob("train/rgbImages/*.png")))
        target_feature_paths = list(sorted(Path(data_path).glob("train/gtLabels/*.png")))
    elif dataset_type == 'val':    
        in_feature_paths = list(sorted(Path(data_path).glob("test/rgbImages/*.png")))
        target_feature_paths = list(sorted(Path(data_path).glob("test/gtLabels/*.png")))
        
    for imgPath, gtPath in zip(in_feature_paths, target_feature_paths):
        img = cv2.imread(str(imgPath))
        gt = cv2.imread(str(gtPath), cv2.IMREAD_GRAYSCALE)
        
        _img = img[:Hcrop, :Wcrop, :]
        _gt = gt[:Hcrop, :Wcrop]
        
        _gtRGB = np.zeros((Hcrop,Wcrop,3))
        _gtNEW = np.zeros_like(_gt)
        
        ## Need to Class 1 to Class 0
        _gtNEW[_gt==1] = 0
        _gtNEW[_gt==2] = 1
        _gtNEW[_gt==3] = 2
        
        _gtRGB[_gtNEW==3] = (255,255,255)
        _gtRGB[_gtNEW==2] = (0,0,255)
        _gtRGB[_gtNEW==1] = (255,0,0)
        #gtRgbOutPath
        cv2.imwrite(gtRgbOutPath+str(imgPath.name), _gtRGB)
        cv2.imwrite(gtOutPath+str(imgPath.name), _gtNEW)
        cv2.imwrite(rgbOutPath+str(imgPath.name), _img)