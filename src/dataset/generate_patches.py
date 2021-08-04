from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='/dataset_sub/camera_light_glare/train_all/', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='/dataset_sub/camera_light_glare/patches_256_all_no_std/train_patch/',type=str, help='Directory for image patches')
#parser.add_argument('--src_dir', default='/dataset_sub/camera_light_glare/val/', type=str, help='Directory for full resolution images')
#parser.add_argument('--tar_dir', default='/dataset_sub/camera_light_glare/patches_768/val_patch/',type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=20, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=20, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores
#STRIDE = int(PS / 2)
STRIDE = PS
#STRIDE = 64

noisy_patchDir = os.path.join(tar, 'train_input_img/')
clean_patchDir = os.path.join(tar, 'train_label_img/')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

#get sorted folders
files_noisy = natsorted(glob(os.path.join(src, 'train_input_img', '*.png')))
files_clean = natsorted(glob(os.path.join(src, 'train_label_img', '*.png')))
noisy_files, clean_files = [], []
for file_ in files_clean:
    filename = os.path.split(file_)[-1]
    clean_files.append(file_)

for file_ in files_noisy:
    filename = os.path.split(file_)[-1]
    noisy_files.append(file_)

def save_files_random(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]
        
        #cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
        #cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)

        std_noisy = noisy_patch.std()
        std_clean = clean_patch.std()
        if noisy_patch.shape[0] >= PS and noisy_patch.shape[1] >= PS and clean_patch.shape[0] >= PS and clean_patch.shape[1] >= PS and std_noisy >= 6.0:
            cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
            cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)

def save_files_stride(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for top in range(0, noisy_img.shape[0], STRIDE):
        for left in range(0, noisy_img.shape[1], STRIDE):
            noisy_patch = noisy_img[top:top + PS, left:left + PS, :]
            clean_patch = clean_img[top:top + PS, left:left + PS, :]

            std_noisy = noisy_patch.std()
            std_clean = clean_patch.std()

            #if noisy_patch.shape[0] >= PS and noisy_patch.shape[1] >= PS and clean_patch.shape[0] >= PS and clean_patch.shape[1] >= PS and std_noisy >= 6.0:
            if noisy_patch.shape[0] >= PS and noisy_patch.shape[1] >= PS and clean_patch.shape[0] >= PS and clean_patch.shape[1] >= PS:
                cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}_{}.png'.format(i,top,left)), noisy_patch)
                cv2.imwrite(os.path.join(clean_patchDir, '{}_{}_{}.png'.format(i,top,left)), clean_patch)
                #cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}_{:.2f}.png'.format(i,left+1,std_noisy)), noisy_patch)
                #cv2.imwrite(os.path.join(clean_patchDir, '{}_{}_{:.2f}.png'.format(i,left+1,std_clean)), clean_patch)

#[save_files_stride(i) for i in tqdm(range(len(noisy_files)))]
#Parallel(n_jobs=NUM_CORES)(delayed(save_files_random)(i) for i in tqdm(range(len(noisy_files))))
Parallel(n_jobs=NUM_CORES)(delayed(save_files_stride)(i) for i in tqdm(range(len(noisy_files))))
