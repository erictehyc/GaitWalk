import os, re, pickle, math, copy, random
import cv2
import numpy as np
import matplotlib.pyplot as plt

SEQ_LEN = 30
# MIN_HEIGHT, MIN_WIDTH = 300, 50
MIN_HEIGHT, MIN_WIDTH = 0, 0


#load data as numpy array. Memory problem here
def load_data(frame_dir, debug=False, seq_size=30):
    X = []
    y = []
    min_h, min_w = math.inf, math.inf
    for fn in sorted(os.listdir(frame_dir)):
        if fn[:3] != "VID":
            continue

        vid_dir = os.path.join(frame_dir, fn)
        id_dirs = [f for f in os.listdir(vid_dir) if re.match(r'id_\d+', f)]
        for id_dirname in id_dirs:
            id_dir = os.path.join(vid_dir, id_dirname)
            seq, seq_min_h, seq_min_w = load_seq(id_dir, seq_size, debug)
            if len(seq) < seq_size:
                print(f'{id_dir} has not enough number of frames detected')
            if seq and len(seq) == seq_size:
                X.append(seq)
                y.append(get_label(fn))
                min_h = min(min_h, seq_min_h)
                min_w = min(min_w, seq_min_w)

    return X, y, min_h, min_w
#Save sequences in X as pickles in a folder for later use in keras generator. (Reduce memory usage by not loading all data at one go during training)
def save_data_as_pickle(frame_dir, pickle_dir, debug=False, seq_size=30):
    print(f"Saving frames from {frame_dir} to {pickle_dir} as pickled data")
    num_seq = 0
    min_h, min_w = math.inf, math.inf
    for fn in sorted(os.listdir(frame_dir)):
    
        if fn[:3] != "VID":
            continue
        vid_dir = os.path.join(frame_dir, fn)
        id_dirs = [f for f in os.listdir(vid_dir) if re.match(r'id_\d+', f)]
        for id_dirname in id_dirs:
            id_dir = os.path.join(vid_dir, id_dirname)
            seq, seq_min_h, seq_min_w = load_seq(id_dir, seq_size, debug)
            if len(seq) < seq_size:
                print(f'{id_dir} has not enough number of frames detected, needed {seq_size}')
            if seq and len(seq) == seq_size:
                pkl_name = f"seq{num_seq:03}_{get_label(fn)}.pkl"
                pkl_path = os.path.join(pickle_dir, pkl_name)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(seq, f)
                min_h = min(min_h, seq_min_h)
                min_w = min(min_w, seq_min_w)
                num_seq += 1

    return num_seq, min_h, min_w

def img_seq_generator(input_path, bs, img_h, img_w, mode='train', shuffle=False):
    filenames = sorted(os.listdir(input_path))
    if shuffle: random.shuffle(filenames)
    filenames = iter(filenames)
    while True:
        X = []
        y = []
        #gets the batches of sequences
        while len(X) < bs:
            pkl_name = next(filenames, None)
            #If already reach the end of directory, loop from first file again.
            if pkl_name is None:
                filenames = sorted(os.listdir(input_path))
                if shuffle: random.shuffle(filenames)
                filenames = iter(filenames)
                if mode == "eval":
                    break
                continue

            pkl_path = os.path.join(input_path, pkl_name)
            with open(pkl_path, 'rb') as f:
                img_seq = pickle.load(f)
            X.append(img_seq)
            y.append(int(pkl_name[-5]))
        if mode == "eval":
            print(y)
        X = normalize_img_data(X, img_h, img_w)
        X = np.array(X)
        y = np.array(y)
        yield (X, y)

def load_predict_data(frame_dir, seq_size=30):
    X = []
    names = []
    for fn in sorted(os.listdir(frame_dir)):
        vid_dir = os.path.join(frame_dir, fn)
        id_dirs = [f for f in os.listdir(vid_dir) if re.match(r'id_\d+', f)]
        for id_dirname in id_dirs:
            id_dir = os.path.join(vid_dir, id_dirname)
            seq, seq_min_h, seq_min_w = load_seq(id_dir, seq_size, False)
            if len(seq) < seq_size:
                print(f'{id_dir} has not enough number of frames detected')
            if seq and len(seq) == seq_size:
                X.append(seq)
                names.append(f'{fn}-{id_dirname}')


    return X, names




#assume img_seq_dir only contains images file ordered with filename
def load_seq(img_seq_dir, seq_size, debug=False):
    min_h, min_w = math.inf, math.inf
    seq = []
    prev_no = None
    try:
        i = 0
        for img_name in sorted(os.listdir(img_seq_dir)):
            #stop when frames are enough
            if i == seq_size:
                break
            #only checks for frame images
            if not re.match(r'frame.*\.jpg', img_name):
                continue
            img_path = os.path.join(img_seq_dir, img_name)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            #make sure image detected is large enough
            if img_array.shape[0] < MIN_HEIGHT or img_array.shape[1] < MIN_WIDTH:
                if debug: print(f'{img_path} contains small image. Please check.')
                # return [], -1, -1
                continue
            #make sure the sequence is not too broken
            cur_no = get_img_frameno(img_name)
            if prev_no and (cur_no - prev_no > 10):
                if debug: print(f'{img_path} contains broken sequence')
                i = 0
                seq = []
            #succesfully added to sequence
            seq.append(img_array)
            i += 1
            prev_no = cur_no
            min_h = min(min_h, img_array.shape[0])
            min_w = min(min_w, img_array.shape[1])
    except NotADirectoryError:
        print(f"Frames for {img_seq_dir} was not extracted correctly")

    return seq, min_h, min_w


#return integer. 1: healthy , 0: depressed
def get_label(vid_name):
    s = r"VID_RGB_\d*_(\d)"
    res = re.match(s, vid_name)
    return int(res.group(1))

def get_img_frameno(img_name):
    s = r"frame(\d+)_id\d+\.jpg"
    res = re.match(s, img_name)
    if not res:
        print(img_name)
    return int(res.group(1))

def normalize_img_data(X, h, w):
    for seq in X:
        for i in range(len(seq)):
            seq[i] = cv2.resize(seq[i], (w, h))
            # show image
            # plt.imshow(seq[i], cmap="gray")
            # plt.show()
            seq[i] = seq[i] / 255
            seq[i] = np.atleast_3d(seq[i])
    return X


if __name__ == "__main__":
    # frame_dir = '/home/jia/jiawen/uni_materials/Y3S2/FYP/frames'
    # frame_dir = '/home/jia/jiawen/uni_materials/Y3S2/FYP/frames/test'
    pickle_dir = 'pickled_seq_images/test'
    # num_seq, min_h, min_w = save_data_as_pickle(frame_dir, pickle_dir, True)
    gen = img_seq_generator(pickle_dir,3, 700, 100, mode='train')
    for i in range(3):
        X , y = next(gen)
        print("At iteration: ", i)
        print(len(y), len(X))

