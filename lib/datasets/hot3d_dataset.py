import sys
import os
import json

sys.path.append(os.path.abspath('.'))

from torch.utils.data import Dataset

from lib.datasets.detect_track_video_hot3d import detect_track_process_gt


def load_test_set(test_file):
    with open(test_file, 'r') as f:
        test_set = json.load(f)
    return test_set

class Hot3dDataset(Dataset):

    def __init__(self, video_root, set_file='val.json', debug_start=None, debug_end=None, preprocess=True, preprocess_for_filling=False,
                 overwrite_track=True, overwrite_process=True, for_eval=False, vis_img_crop_j2d=False, test_name=None):
        super(Hot3dDataset, self).__init__()

        self.video_root = video_root
        if not test_name is None:
            self.set = [test_name]
            self.test_video = True
        else:
            self.set = load_test_set(os.path.join(self.video_root, set_file))
            self.test_video = False
        self.crop_size=256
        self.overwrite_track = overwrite_track
        self.overwrite = overwrite_process
        self.for_eval = for_eval
        self.vis_img_crop_j2d = vis_img_crop_j2d
        if not debug_start is None:
            self.set = self.set[debug_start:debug_end]
        if preprocess:
            self.preprocess()

    def preprocess(self):
        item_idx = 0
        cnt = 0
        seq_num = 0
        for video in self.set:
            cnt += 1
            items_all = []
            print(f"processing {cnt}/{len(self.set)} videos")
            video_path = os.path.join(self.video_root, video, video+'.mp4')
            seq_folder = os.path.join(self.video_root, video, 'preprocess')
            if self.overwrite_track or not os.path.exists(f'{seq_folder}/preprocess_gtdet_tracks.npy'):
                with open(os.path.join(self.video_root, video, 'focal.txt'), 'r') as file:
                    focal_length = file.read()
                    img_focal = float(focal_length)
                img_folder = detect_track_process_gt(video_path, save_vos=False, 
                                                          video_root=self.video_root, video=video, img_focal=img_focal, test_video=self.test_video)
            
    
