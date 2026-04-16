import argparse
import sys
import os

sys.path.append(os.path.abspath('.'))

from lib.datasets.hot3d_dataset import Hot3dDataset
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, default='')
    parser.add_argument("--set_file", type=str, default='val.json')
    parser.add_argument("--vis_img_crop_j2d", action="store_true")
    parser.add_argument("--test_name", type=str, default=None)
    parser.add_argument("--for_eval", action="store_true")
    args = parser.parse_args()

    dataset = Hot3dDataset(args.video_root, args.set_file, preprocess=True, vis_img_crop_j2d=args.vis_img_crop_j2d, test_name=args.test_name, for_eval=args.for_eval)
