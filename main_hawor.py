import os
import sys
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import interp1d

# PyTorch 2.6+ weights_only security fix
try:
    import functools
    _orig_load = torch.load
    @functools.wraps(_orig_load)
    def _patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except:
    pass

# NumPy 1.24+ compatibility fix for old dependencies
import numpy as np
for name in ['bool', 'int', 'float', 'complex', 'object', 'str']:
    if not hasattr(np, name):
        setattr(np, name, getattr(np, f"{name}_" if name != 'str' else 'str_'))
if not hasattr(np, 'unicode'):
    setattr(np, 'unicode', np.str_)

# Relative imports
from lib.models.hawor import HAWOR
from hawor.configs import get_config
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.pipeline.tools import parse_chunks
from ultralytics import YOLO

def load_hawor(checkpoint_path):
    """
    1. Loads the config from model_config.yaml.
    2. Corrects the BBOX_SHAPE to [192, 256] if the backbone is vit.
    3. Returns the initialized HAWOR model in eval() mode.
    """
    model_cfg_path = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=False)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    model = model.to(device)
    model.eval()
    return model, model_cfg

def interpolate_bboxes(bboxes):
    non_zero_indices = np.where(np.any(bboxes != 0, axis=1))[0]
    if len(non_zero_indices) < 2:
        return bboxes
    
    zero_indices = np.where(np.all(bboxes == 0, axis=1))[0]
    interpolated_bboxes = bboxes.copy()
    for i in range(bboxes.shape[1]):
        interp_func = interp1d(non_zero_indices, bboxes[non_zero_indices, i], kind='linear', fill_value="extrapolate")
        interpolated_bboxes[zero_indices, i] = interp_func(zero_indices)
    return interpolated_bboxes

def detect_track(frames, model_path='./weights/external/detector.pt', thresh=0.2):
    """
    3. Include a detect_track(frames, model, thresh) function using ultralytics.YOLO to track hands
       across frames. It should return a dictionary of tracks.
    """
    hand_det_model = YOLO(model_path)
    tracks = {}
    
    for t, imgpath in enumerate(tqdm(frames, desc="Detecting Hands")):
        img_cv2 = cv2.imread(imgpath)
        with torch.no_grad():
            results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(boxes)

            boxes_with_conf = np.hstack([boxes, confs[:, None]])
            
            for i in range(len(boxes)):
                tid = track_ids[i]
                if tid == -1: continue # Skip untracked detections
                
                subj = {
                    'frame': t,
                    'det': True,
                    'det_box': boxes_with_conf[[i]],
                    'det_handedness': cls[[i]] # 0: Left, 1: Right (usually)
                }
                
                if tid not in tracks:
                    tracks[tid] = []
                tracks[tid].append(subj)
                
    return tracks

def hawor_motion_estimation(frames, tracks, model, img_focal=600):
    """
    4. Include a hawor_motion_estimation function that:
       - Handles bounding box interpolation (interpolate_bboxes).
       - Handles chunking of frames (parse_chunks).
       - Calls model.inference(...) for each chunk.
       - Crucial: Implements the left-hand flipping logic.
    """
    img = cv2.imread(frames[0])
    img_center = [img.shape[1] / 2, img.shape[0] / 2]
    
    final_output = {'left': {}, 'right': {}}
    
    for tid, trk in tracks.items():
        # Interpolate
        valid_frames = np.array([t['frame'] for t in trk])
        boxes = np.concatenate([t['det_box'] for t in trk])
        
        # Determine handedness for this track
        is_right_votes = np.concatenate([t['det_handedness'] for t in trk])
        is_right = (is_right_votes.sum() / len(is_right_votes)) >= 0.5
        handedness = 'right' if is_right else 'left'
        do_flip = not is_right
        
        # Prepare full sequence for interpolation
        full_boxes = np.zeros((len(frames), 5))
        full_boxes[valid_frames] = boxes
        
        start_f, end_f = valid_frames[0], valid_frames[-1]
        full_boxes[start_f:end_f+1] = interpolate_bboxes(full_boxes[start_f:end_f+1])
        
        # Chunking
        chunk_frames, chunk_boxes = parse_chunks(np.arange(start_f, end_f+1), full_boxes[start_f:end_f+1], min_len=1)
        
        for f_ck, b_ck in zip(chunk_frames, chunk_boxes):
            img_ck = [frames[f] for f in f_ck]
            
            with torch.no_grad():
                results = model.inference(img_ck, b_ck, img_focal=img_focal, img_center=img_center, do_flip=do_flip)
            
            # Extract params
            pred_rotmat = results["pred_rotmat"] # (T, 16, 3, 3)
            pred_trans = results["pred_trans"]   # (T, 3)
            pred_shape = results["pred_shape"]   # (10)
            pred_kpts = results["pred_keypoints_2d"] # (T, 21, 2)
            
            # Convert rotmat to angle-axis
            pred_aa = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 16, 3)
            
            for i, frame_idx in enumerate(f_ck):
                root_orient = pred_aa[i, 0].cpu().numpy()
                hand_pose = pred_aa[i, 1:].cpu().numpy().flatten()
                transl = pred_trans[i].cpu().numpy()
                beta = pred_shape.cpu().numpy()
                kpts = pred_kpts[i].cpu().numpy()

                # Left-hand flipping logic
                if do_flip:
                    root_orient[1] *= -1
                    root_orient[2] *= -1
                    hand_pose_reshaped = hand_pose.reshape(15, 3)
                    hand_pose_reshaped[:, 1] *= -1
                    hand_pose_reshaped[:, 2] *= -1
                    hand_pose = hand_pose_reshaped.flatten()
                    
                    # Flip keypoints x-axis
                    img_w = img_center[0] * 2
                    kpts[:, 0] = img_w - 1 - kpts[:, 0]

                final_output[handedness][int(frame_idx)] = {
                    'beta': beta.tolist(),
                    'hand_pose': hand_pose.tolist(),
                    'global_orient': root_orient.tolist(),
                    'transl': transl.tolist(),
                    'keypoints_2d': kpts.tolist()
                }
                
    return final_output

def recon(frames, focal_length=600):
    """
    5. The recon(frames, focal_length) method must return a dictionary:
    """
    checkpoint = './weights/hawor/checkpoints/hawor.ckpt'
    model, _ = load_hawor(checkpoint)
    
    tracks = detect_track(frames)
    
    results = hawor_motion_estimation(frames, tracks, model, img_focal=focal_length)
    return results

if __name__ == "__main__":
    import argparse
    from glob import glob
    from natsort import natsorted
    
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file or image folder")
    parser.add_argument("--output", default="results_hawor.json", help="Output JSON file")
    args = parser.parse_args()
    
    # Extract frames if video
    if os.path.isfile(args.video):
        video_name = Path(args.video).stem
        temp_dir = f"temp_{video_name}"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Extracting frames to {temp_dir}...")
        os.system(f"ffmpeg -i {args.video} -vf fps=30 -q:v 2 {temp_dir}/%04d.jpg -y")
        frames = natsorted(glob(f"{temp_dir}/*.jpg"))
    else:
        frames = natsorted(glob(f"{args.video}/*.jpg"))
        
    if not frames:
        print("No frames found.")
        sys.exit(1)
        
    results = recon(frames)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {args.output}")
