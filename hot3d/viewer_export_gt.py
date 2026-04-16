# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import subprocess
from typing import Optional, Type

import rerun as rr  # @manual
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import loadManoHandModel
from data_loaders.hamer_mano_layer import loadHamerManoHandModel

import sys
sys.path.append("..")
import pickle
import numpy as np
from collections import defaultdict
import torch
import joblib

try:
    from dataset_api import Hot3dDataProvider  # @manual
except ImportError:
    from hot3d.dataset_api import Hot3dDataProvider

try:
    from Hot3DVisualizer import Hot3DVisualizer
except ImportError:
    from hot3d.Hot3DVisualizer import Hot3DVisualizer

from tqdm import tqdm
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="path to hot3d data sequence",
        required=True,
    )
    parser.add_argument(
        "--object_library_folder",
        type=str,
        help="path to object library folder containing instance.json and *.glb cad files",
        required=True,
    )
    parser.add_argument(
        "--mano_model_folder",
        type=str,
        default=None,
        help="path to MANO models containing the MANO_RIGHT/LEFT.pkl files",
        required=False,
    )

    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)

    # If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument(
        "--rrd_output_path", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--use_slam_hamer", action="store_true"
    )

    return parser.parse_args()


def execute_rerun(
    sequence_folder: str,
    object_library_folder: str,
    mano_model_folder: Optional[str],
    rrd_output_path: Optional[str],
    jpeg_quality: int,
    timestamps_slice: Type[slice],
    fail_on_missing_data: bool,
    use_slam_hamer: bool,
):
    export_gt_folder = sequence_folder.replace('dataset', 'hot3d_dataset_export')
    if not os.path.exists(export_gt_folder):
        os.makedirs(export_gt_folder)
    # if os.path.exists(os.path.join(export_gt_folder, export_gt_folder.split('/')[-1] + '.mp4')):
    #     return

    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(object_library_folder):
        raise RuntimeError(
            f"Object Library folder {object_library_folder} does not exist"
        )

    object_library = load_object_library(
        object_library_folderpath=object_library_folder
    )

    if use_slam_hamer:
        mano_hand_model = loadHamerManoHandModel(mano_model_folder)
    else:
        mano_hand_model = loadManoHandModel(mano_model_folder)

    #
    # Initialize hot3d data provider
    #
    data_provider = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        mano_hand_model=mano_hand_model,
        fail_on_missing_data=fail_on_missing_data,
        use_slam_hamer=use_slam_hamer,
    )
    print(f"data_provider statistics: {data_provider.get_data_statistics()}")

    #
    # Prepare the rerun rerun log configuration
    #
    # rr.init("hot3d Data Viewer", spawn=(rrd_output_path is None))
    # if rrd_output_path is not None:
    #     print(f"Saving .rrd file to {rrd_output_path}")
    #     rr.save(rrd_output_path)

    #
    # Initialize the rerun hot3d visualizer interface
    #
    rr_visualizer = Hot3DVisualizer(data_provider)

    # Define which image stream will be shown
    image_stream_ids = data_provider.device_data_provider.get_image_stream_ids()

    #
    # Log static assets (aka Timeless assets)
    rr_visualizer.log_static_assets(image_stream_ids, disable_vis=True)

    timestamps = data_provider.device_data_provider.get_sequence_timestamps()
    #
    # Visualize the dataset sequence
    #
    # Loop over the timestamps of the sequence and visualize corresponding data
    image_list = []
    tt_list = []
    ego_extrinsics_list = []
    head_pose_list = []
    data_video = defaultdict(list)

    for timestamp in tqdm(timestamps[timestamps_slice]):

        rr.set_time_nanos("synchronization_time", int(timestamp))
        rr.set_time_sequence("timestamp", timestamp)

        ego_image_cv2, head_pose, ego_extrinsics, ego_focal, data_img = rr_visualizer.log_dynamic_assets(image_stream_ids, timestamp, disable_vis=True)
        if head_pose is None:
            continue
        for key in data_img.keys():
            data_video[key].extend(data_img[key])
        ego_extrinsics_list.append(ego_extrinsics)
        head_pose_list.append(head_pose)
        image_list.append(ego_image_cv2)
        tt_list.append(timestamp)
    data_video_all = {}
    for key in data_video.keys():
        if torch.is_tensor(data_video[key][0]):
            data_video_all[key] = torch.stack(data_video[key])
    save_pth = os.path.join(export_gt_folder, 'anno.pth')
    joblib.dump(data_video_all, save_pth)
    print("saved", save_pth)
    head_pose_list = np.stack(head_pose_list)
    ego_extrinsics_list = np.stack(ego_extrinsics_list)
    with open(os.path.join(export_gt_folder, 'head_pose.pkl'), 'wb') as file:
        pickle.dump(head_pose_list, file)
    with open(os.path.join(export_gt_folder, 'ego_extrinsics.pkl'), 'wb') as file:
        pickle.dump(ego_extrinsics_list, file)
    with open(os.path.join(export_gt_folder, 'focal.txt'), 'w') as file:
        file.write(str(ego_focal))
    
    with open(os.path.join(export_gt_folder, 'tt_list.pkl'), 'wb') as file:
        pickle.dump(tt_list, file)
    create_video_from_list_fast(image_list, os.path.join(export_gt_folder, export_gt_folder.split('/')[-1] + '.mp4'))


def create_video_from_list_fast(image_list, output_path):
    tmp_folder = os.path.join(os.path.dirname(output_path), 'extracted_images')
    os.makedirs(tmp_folder, exist_ok=True)
    for t, img in enumerate(tqdm(image_list)):
        cv2.imwrite(os.path.join(tmp_folder, f"{t:04d}.jpg"), img)
    create_video_from_images_ffmpeg(tmp_folder, output_path)
    # shutil.rmtree(tmp_folder)
    

def create_video_from_images_ffmpeg(image_folder, output_video, frame_rate=30):
    # 确保图片文件夹存在
    if not os.path.isdir(image_folder):
        raise ValueError(f"Image folder {image_folder} does not exist.")
    
    # 构建 FFmpeg 命令
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(frame_rate),
        '-i', os.path.join(image_folder, '%04d.jpg'),  # 假设图片文件名为 0001.png, 0002.png, 0003.png, ...
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    # 打印 FFmpeg 命令（可选）
    print('Running FFmpeg command:', ' '.join(ffmpeg_command))
    
    # 调用 FFmpeg 命令
    subprocess.run(ffmpeg_command, check=True)

def create_video_from_list(image_list, output_path, fps=30):
    # Check if the list is not empty
    if not image_list:
        print("The image list is empty.")
        return
    
    # Get the width and height of the images
    height, width, _ = image_list[0].shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame to the video file
    for image in tqdm(image_list):
        video_writer.write(image)
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_path}")

def main():
    args = parse_args()
    print(f"args provided: {args}")

    try:
        execute_rerun(
            sequence_folder=args.sequence_folder,
            object_library_folder=args.object_library_folder,
            mano_model_folder=args.mano_model_folder,
            rrd_output_path=args.rrd_output_path,
            jpeg_quality=args.jpeg_quality,
            timestamps_slice=slice(None, 300, None),
            fail_on_missing_data=False,
            use_slam_hamer=args.use_slam_hamer,
        )
    except Exception as error:
        print(f"An exception occurred: {error}")

def export_gt(sequence_folder, start_frame=20, debug=False):
    print(sequence_folder)
    # args = parse_args()
    # print(f"args provided: {args}")
    from easydict import EasyDict as edict
    args = edict()
    args.object_library_folder = "dataset/assets"
    args.mano_model_folder = "mano_v1_2/models/"
    args.rrd_output_path = None
    args.jpeg_quality = 75
    args.use_slam_hamer = False

    if debug:
        len = 300
    else:
        len = None

    execute_rerun(
        sequence_folder=sequence_folder,
        object_library_folder=args.object_library_folder,
        mano_model_folder=args.mano_model_folder,
        rrd_output_path=args.rrd_output_path,
        jpeg_quality=args.jpeg_quality,
        timestamps_slice=slice(start_frame, len, None),
        fail_on_missing_data=False,
        use_slam_hamer=args.use_slam_hamer,
    )


if __name__ == "__main__":
    main()
