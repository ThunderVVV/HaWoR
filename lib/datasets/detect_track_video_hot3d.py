import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
from glob import glob
from lib.pipeline.tools import detect_track_gt

def detect_track_process_gt(file, save_vos=True, video_root=None, video=None, img_focal=None, K_matrix=None, test_video=False, dataset_type='hot3d', fix_shapedirs=True):
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = f'{root}/preprocess'
    img_folder = f'{seq_folder}/preprocess_images'
    os.makedirs(seq_folder, exist_ok=True)
    print(f'Running on {file} ...')

    ##### Extract Frames #####
    img_folder = f"{root}/extracted_images"

    ##### Detection + SAM + DEVA-Track-Anything #####
    print('Detect, Segment, and Track ...')
    # save_vos = True
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        imgfiles = sorted(glob(f'{img_folder}/*.png'))
    boxes_, tracks_ = detect_track_gt(imgfiles, seq_folder, thresh=0.25, 
                                                min_size=None, save_vos=save_vos,
                                                video_root=video_root, video=video, img_focal=img_focal, K_matrix=K_matrix, test_video=test_video,
                                                dataset_type=dataset_type, fix_shapedirs=fix_shapedirs)
    np.save(f'{seq_folder}/preprocess_gtdet_tracks.npy', tracks_)
    return img_folder


