import numpy as np
import glob2
import math
import cv2
from pathlib import Path

data_path = '/dataset/Woodscape/soiling_dataset_nodist_50/'
# Hcrop = 736
# Wcrop = 992
Hcrop = 224
Wcrop = 320

dataset_type = 'val'

gtOutPath = data_path + 'test/gtLabels_crop/'
rgbOutPath = data_path + 'test/rgbImages_crop/'

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
        
        cv2.imwrite(gtOutPath+str(imgPath.name), _gt)
        cv2.imwrite(rgbOutPath+str(imgPath.name), _img)