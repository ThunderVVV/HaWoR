import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import math
import argparse
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from pycocotools import mask as masktool
from lib.pipeline.masked_droid_slam import *
from lib.pipeline.est_scale import *
from lib.datasets.hot3d_dataset import load_test_set
from hawor.utils.process import block_print, enable_print

sys.path.insert(0, os.path.dirname(__file__) + '/../../thirdparty/Metric3D')
from metric import Metric3D

def get_all_mp4_files(folder_path):
    # Ensure the folder path is absolute
    folder_path = os.path.abspath(folder_path)
    
    # Recursively search for all .mp4 files in the folder and its subfolders
    mp4_files = glob(os.path.join(folder_path, '**', '*.mp4'), recursive=True)
    
    return mp4_files

def split_list_by_interval(lst, interval=1000):
    start_indices = []
    end_indices = []
    split_lists = []
    
    for i in range(0, len(lst), interval):
        start_indices.append(i)
        end_indices.append(min(i + interval, len(lst)))
        split_lists.append(lst[i:i + interval])
    
    return start_indices, end_indices, split_lists


parser = argparse.ArgumentParser()
parser.add_argument("--img_focal", type=float)
parser.add_argument('--img_center', nargs='+', default=None)
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument("--set_file", type=str, default='val.json')
parser.add_argument("--video_root", type=str, default='datasets/hot3d_valset_export')
parser.add_argument("--img_folder", type=str, default='extracted_images')
args = parser.parse_args()

# File and folders
video_root = args.video_root
testset = load_test_set(os.path.join(video_root, args.set_file))
for video in testset:
    seq_folder = os.path.join(video_root, video, 'preprocess')
    video_folder = os.path.join(video_root, video)

    img_folder = f'{video_folder}/{args.img_folder}'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    start_idxs, end_idxs, imgfiles_list = split_list_by_interval(imgfiles)
    for part, imgfiles in enumerate(imgfiles_list):

        if os.path.exists(f'{seq_folder}/hawor_slam_w_scale_{start_idxs[part]}_{end_idxs[part]}.npz'):
            print(f"skip {seq_folder}/hawor_slam_w_scale_{start_idxs[part]}_{end_idxs[part]}")
            continue

        first_img = cv2.imread(imgfiles[0])
        height, width, _ = first_img.shape
        with open(os.path.join(video_folder, 'focal.txt'), 'r') as file:
            focal_length = file.read()
            focal_length = float(focal_length)
        args.img_focal = focal_length
        print(f'Running on {video_folder} ...')

        ##### Run Masked DROID-SLAM #####
        # Use Masking
        masks = np.load(f'{seq_folder}/model_masks.npy', allow_pickle=True)
        masks = masks[start_idxs[part]:end_idxs[part]]
        masks = np.array([masktool.decode(m) for m in masks])
        masks = torch.from_numpy(masks)

        # Camera calibration (intrinsics) for SLAM
        focal = args.img_focal
        calib = np.array(est_calib(imgfiles))
        center = calib[2:]        
        calib[:2] = focal
        
        # Droid-slam with masking
        droid, traj = run_slam(imgfiles, masks=masks, calib=calib)
        n = droid.video.counter.value
        tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
        disps = droid.video.disps_up.cpu().numpy()[:n]
        print('DBA errors:', droid.backend.errors)

        del droid
        torch.cuda.empty_cache()

        # Estimate scale  
        block_print()  
        metric = Metric3D('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth') 
        enable_print() 
        min_threshold = 0.4
        max_threshold = 0.7

        print('Predicting Metric Depth ...')
        pred_depths = []
        H, W = get_dimention(imgfiles)
        for t in tqdm(tstamp):
            pred_depth = metric(imgfiles[t], calib)
            pred_depth = cv2.resize(pred_depth, (W, H))
            pred_depths.append(pred_depth)

        ##### Estimate Metric Scale #####
        print('Estimating Metric Scale ...')
        scales_ = []
        n = len(tstamp)   # for each keyframe
        for i in tqdm(range(n)):
            t = tstamp[i]
            disp = disps[i]
            pred_depth = pred_depths[i]
            slam_depth = 1/disp
            
            # Estimate scene scale
            msk = masks[t].numpy().astype(np.uint8)
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)  
            while math.isnan(scale):
                min_threshold -= 0.1
                max_threshold += 0.1
                scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)                    
            scales_.append(scale)

        median_s = np.median(scales_)
        print(f"estimated scale: {median_s}")

        # Save results
        os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
        save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idxs[part]}_{end_idxs[part]}.npz'
        np.savez(save_path, 
                tstamp=tstamp, disps=disps, traj=traj, 
                img_focal=focal, img_center=calib[-2:],
                scale=median_s)  