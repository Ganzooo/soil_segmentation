#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch
import cv2
import torchvision
from augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale, RandomVerticallyFlip
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_dataset(data_path, batch_size, distributed, center_crop=False, random_crop=False, resize_size=(512,512), model_type='baseline', color_domain='rgb'):

    transformer_train = A.Compose([
            A.Resize(resize_size[0],resize_size[1]),
            #A.RandomCrop(width=resize_size[0], height=resize_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
            ])
    transformer_valid = A.Compose([
            A.Resize(resize_size[0],resize_size[1]),
            ToTensorV2()
            ])

    train_dataset = DataLoaderImg(data_path, mode='train', resize_size=resize_size, transform=transformer_train, color_domain=color_domain)
    val_dataset = DataLoaderImg(data_path, mode='val', resize_size=resize_size, transform=transformer_valid, color_domain=color_domain)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)
    
    return train_dataloader, val_dataloader

class DataLoaderImg(Dataset):
    def __init__(self, data_path, mode='train', transform=True, resize_size=(512,512), color_domain='rgb'):
        self.mode = mode
        self.data_path = data_path
        self.transform = transform
        self.target_size = resize_size
        self.dataset_type = mode
        self.color_domain = color_domain
        augmentations = Compose([RandomHorizontallyFlip(0.5), RandomVerticallyFlip(0.5)])
        self.augmentations = augmentations

        # if self.dataset_type == 'train':
        #     self.in_feature_paths = list(sorted(Path(self.data_path).glob("train/rgbImages/*.png")))
        #     self.target_feature_paths = list(sorted(Path(self.data_path).glob("train/gtLabels/*.png")))
        # elif self.dataset_type == 'val':    
        #     self.in_feature_paths = list(sorted(Path(self.data_path).glob("test/rgbImages/*.png")))
        #     self.target_feature_paths = list(sorted(Path(self.data_path).glob("test/gtLabels/*.png")))
            
        if self.dataset_type == 'train':
            self.in_feature_paths = list(sorted(Path(self.data_path).glob("train/rgbImages_crop/*.png")))
            self.target_feature_paths = list(sorted(Path(self.data_path).glob("train/gtLabels_crop/*.png")))
        elif self.dataset_type == 'val':    
            self.in_feature_paths = list(sorted(Path(self.data_path).glob("test/rgbImages_crop/*.png")))
            self.target_feature_paths = list(sorted(Path(self.data_path).glob("test/gtLabels_crop/*.png")))

    def __len__(self):
        return len(self.in_feature_paths)
    
    def __transform__(self, img):        
        img = cv2.resize(img, (self.target_size[0], self.target_size[1]))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

    def __getitem__(self, idx):
        if self.color_domain == 'ycbcr':
            img = cv2.cvtColor(cv2.imread(str(self.in_feature_paths[idx])), cv2.COLOR_BGR2YCR_CB)
            lbl = cv2.cvtColor(cv2.imread(str(self.target_feature_paths[idx])), cv2.COLOR_BGR2YCR_CB)
        else: 
            img = cv2.cvtColor(cv2.imread(str(self.in_feature_paths[idx])), cv2.COLOR_BGR2RGB)
            #lbl = cv2.imread(str(self.target_feature_paths[idx]), cv2.IMREAD_GRAYSCALE)
            lbl = cv2.cvtColor(cv2.imread(str(self.target_feature_paths[idx])), cv2.COLOR_BGR2GRAY)
        #if self.augmentations is not None:
        #    img, lbl = self.augmentations(img, lbl)
        lbl = torch.from_numpy(lbl).long()
        return self.__transform__(img), lbl, self.in_feature_paths[idx].name
        # data = self.transform(image=img, mask=lbl)
        # img = (data['image'] / 255.0)
        # lbl = (data['mask'] / 255.0)
        # lbl = lbl.permute(2, 0, 1)
        # return img, lbl, self.in_feature_paths[idx].name