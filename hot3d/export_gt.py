import pickle
from viewer_export_gt import export_gt
import json
# import concurrent.futures


def load_set(set_file='train_set.pkl'):
    try:
        with open(set_file, 'rb') as f:
            set = pickle.load(f)
        return set
    except FileNotFoundError:
        return f"Error: The file {set_file} does not exist."

if __name__ == "__main__":
    
    set = ['P0001_a68492d5', 'P0001_9b6feab7', 'P0014_8254f925', 'P0011_76ea6d47', 'P0014_84ea2dcc', 'P0001_8d136980', 'P0012_476bae57', 'P0012_130a66e1', 'P0014_24cb3bf0', 'P0010_1c9fe708', 'P0002_2ea9af5b', 'P0011_11475e24', 'P0010_0ecbf39f', 'P0010_160e551c', 'P0015_42b8b389', 'P0012_915e71c6', 'P0002_65085bfc', 'P0011_47878e48', 'P0011_cee8fe4f', 'P0002_016222d1', 'P0012_d85e10f6', 'P0012_119de519', 'P0010_41c4c626', 'P0012_f7e3880b', 'P0009_02511c2f', 'P0011_72efb935', 'P0010_924e574e']
    print(set)
    cnt = 0
    for video_name in set:
        export_gt("dataset/" + video_name, start_frame=20)
        cnt += 1
        print(f"finished {cnt}/{len(set)} test videos")
    with open('hot3d_dataset_export/val.json', 'w') as f:
        json.dump(set, f)