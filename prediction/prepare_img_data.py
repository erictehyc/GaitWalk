import os, re, math, cv2, sys, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = os.getcwd()
called_dir = BASE_DIR
while os.path.basename(BASE_DIR) != "fyp_team4c":
    path = Path(BASE_DIR)
    BASE_DIR = str(path.parent)
    if BASE_DIR == '/':
        print("Please call this script in the fyp_team4c directory")
        break
sys.path.append(BASE_DIR)
from utils import *
logger = create_logger()
SEQ_LEN = 30
MIN_HEIGHT, MIN_WIDTH = 300, 50
logger.info(f"Prediction sequence length: {SEQ_LEN}, Min. img size: {MIN_WIDTH}x{MIN_HEIGHT}")
logger.warning("Make sure the sequence length fits architecture of model")
# def load_data(frame_dir, debug=False):
#     X = []
#     y = []
#     min_h, min_w = math.inf, math.inf
#     for fn in sorted(os.listdir(frame_dir)):
#         if fn[:3] != "VID":
#             continue

#         vid_dir = os.path.join(frame_dir, fn)
#         id_dirs = [f for f in os.listdir(vid_dir) if re.match(r'id_\d+', f)]
#         for id_dirname in id_dirs:
#             id_dir = os.path.join(vid_dir, id_dirname)
#             seq, seq_min_h, seq_min_w = load_seq(id_dir, SEQ_LEN, debug)
#             if len(seq) < SEQ_LEN:
#                 print(f'{id_dir} has not enough number of frames detected')
#             if seq and len(seq) == SEQ_LEN:
#                 X.append(seq)
#                 y.append(get_label(fn))
#                 min_h = min(min_h, seq_min_h)
#                 min_w = min(min_w, seq_min_w)

#     return X, y, min_h, min_w

# def load_predict_data_frames(frame_dir):
#     X = []
#     names = []
#     for fn in sorted(os.listdir(frame_dir)):
#         vid_dir = os.path.join(frame_dir, fn)
#         id_dirs = [f for f in os.listdir(vid_dir) if re.match(r'id_\d+', f)]
#         for id_dirname in id_dirs:
#             id_dir = os.path.join(vid_dir, id_dirname)
#             seq, seq_min_h, seq_min_w = load_seq(id_dir, SEQ_LEN, False)
#             if len(seq) < SEQ_LEN:
#                 print(f'{id_dir} has not enough number of frames detected')
#             if seq and len(seq) == SEQ_LEN:
#                 X.append(seq)
#                 names.append(f'{fn}-{id_dirname}')


#     return X, names

def load_predict_data_vid(vid_dir, fn, debug=False):
    X = []
    names = []
    images = []
    id_dirs = [f for f in os.listdir(vid_dir) if re.match(r'id_\d+', f)]
    for id_dirname in id_dirs:
        id_dir = os.path.join(vid_dir, id_dirname)
        seq, seq_min_h, seq_min_w = load_seq(id_dir, SEQ_LEN, debug)
        print(seq_min_h, seq_min_w)
        if len(seq) < SEQ_LEN:
            if debug: logger.debug(f'{id_dir} has not enough number of frames detected')
        if seq and len(seq) == SEQ_LEN:
            X.append(seq)
            img_name = os.listdir(id_dir)[0]
            img_path = os.path.join(id_dir, img_name)
            images.append(img_path)
            names.append(f'{fn}-{id_dirname}')
    if len(X) == 0: 
        logger.error(f"Could not collect sequential frames from input video: {vid_dir}")
                
    return X, names, images


#assume img_seq_dir only contains images file ordered with filename
def load_seq(img_seq_dir, seq_len, debug=False):
    min_h, min_w = math.inf, math.inf
    seq = []
    prev_no = None
    try:
        i = 0
        for img_name in sorted(os.listdir(img_seq_dir)):
            #stop when frames are enough
            if i == SEQ_LEN:
                break
            #only checks for frame images
            if not re.match(r'frame.*\.jpg', img_name):
                continue
            img_path = os.path.join(img_seq_dir, img_name)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            #make sure image detected is large enough
            if img_array.shape[0] < MIN_HEIGHT or img_array.shape[1] < MIN_WIDTH:
                if debug: logger.debug(f'{img_path} contains small image. Please check.')
                # return [], -1, -1
                continue
            #make sure the sequence is not too broken
            cur_no = get_img_frameno(img_name)
            if prev_no and (cur_no - prev_no > 10):
                if debug: logger.debug(f'{img_path} contains broken sequence')
                i = 0
                seq = []
            #succesfully added to sequence
            seq.append(img_array)
            i += 1
            prev_no = cur_no
            min_h = min(min_h, img_array.shape[0])
            min_w = min(min_w, img_array.shape[1])
    except NotADirectoryError:
        logger.error(f"Frames for {img_seq_dir} was not extracted correctly. There are no frames directory detected")

    return seq, min_h, min_w


#return integer. 1: healthy , 0: depressed
def get_label(vid_name):
    s = r"VID_RGB_\d*_(\d)"
    res = re.match(s, vid_name)
    if not res:
        logger.warning(f"Could not find a label for {vid_name}. Note that video have to be named in format: VID_RGB_XXX_Y where XXX is the ID and Y is the label")
    return int(res.group(1))

def get_img_frameno(img_name):
    s = r"frame(\d+)_id\d+\.jpg"
    res = re.match(s, img_name)
    if not res:
        logger.warning(f"Could not frame number for {img_name}. Note that frame iamges have to be named in format:frameYYY_idXX.jpg where Y is the frame number")
    return int(res.group(1))

def normalize_img_data(X, h, w):
    for seq in X:
        for i in range(len(seq)):
            seq[i] = cv2.resize(seq[i], (w, h))
            seq[i] = seq[i] / 255
            seq[i] = np.atleast_3d(seq[i])
    return X


if __name__ == "__main__":
    vid_dir = "/home/jia/jiawen/uni_materials/Y3S2/FYP/fyp_virtualenv/prediction/output/frames/pl_sad"
    fn = "pl_sad"
    X, names = load_predict_data_vid(vid_dir, fn, debug=True)
    X = normalize_img_data(X, MIN_HEIGHT, MIN_WIDTH)
    X = np.array(X)
    print(X[0,].shape)


