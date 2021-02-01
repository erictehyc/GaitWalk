import os, sys, pickle, json
from pathlib import Path
BASE_DIR = os.getcwd()
called_dir = BASE_DIR
while os.path.basename(BASE_DIR) != "fyp_team4c":
    path = Path(BASE_DIR)
    BASE_DIR = str(path.parent)
    if BASE_DIR == '/':
        print("Please call this script in the fyp_team4c directory")
        break
TEST_PATH = os.path.join(BASE_DIR, 'testing', 'tracking')
sys.path.append(BASE_DIR)
from utils import *
FIELD1 = "Number of frames detected"
FIELD2 = "First frame appearance"
FIELD3 = "Last frame appearance"
FIELD4 = "Number of broken sequence"
FIELD5 = "Number of missing frames"
#output a json file with this format
"""
{
    video_filename1 : {
        id1: {
            "Number of frames detected": xxx,
            "First frame appearance": xxx,
            "Last frame appearance": xxx,
            "Number of broken sequence": xxx
        },
        id2: ...
    }
    video_filename2:...
}

"""

def track(input_dir, output_dir):
    #perform deepsort on input videos and output result in output folder
    vid_l = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".mp4")]
    for vid_name in vid_l:
        video_fp = os.path.join(input_dir, vid_name)
        run_deepsort(output_dir, video_fp, True, called_dir, BASE_DIR)

    #Extract frames to frames folder
    frames_folder = os.path.join(output_dir, 'frames')
    if not os.path.exists(frames_folder):
        # get_ipython().system('cd OUTPUT_DIR && mkdir frames')
        os.makedirs(frames_folder)
    for vid_name in vid_l:
        print('Extracting: ',vid_name)
        video_fp = os.path.join(input_dir, vid_name)
        extract_frames(video_fp, vid_name, frames_folder)

    
    # In[8]:
    #Extract bbox of the frames
    bbox_dir = os.path.join(output_dir, 'bbox_output')
    for vid_name in vid_l:
        pickle_path = os.path.join(bbox_dir, f'{vid_name[:-4]}_bbox.pkl')
        print("Extracting ", pickle_path)
        extract_bbox(vid_name[:-4], pickle_path, frames_folder)

    
def get_track_info(bbox_dir):
    bbox_l = [f for f in sorted(os.listdir(bbox_dir)) if f.endswith(".pkl")]
    track_info = {}
    for fn in bbox_l:
        vid_track_info = {}
        bbox_path = os.path.join(bbox_dir, fn)
        print(bbox_path)
        with open(bbox_path, 'rb') as infile:
            saved_bbox_info =  pickle.load(infile)

        for framenum_and_box in saved_bbox_info:
            frame_num, identities, bboxs = framenum_and_box
            for id_num in identities:
                id_num = f"id_{id_num:03d}"
                if vid_track_info.get(id_num) is None:
                    vid_track_info[id_num] = {
                        FIELD1: 1,
                        FIELD2: frame_num,
                        FIELD3: frame_num,
                        FIELD4: 0
                    }
                else:
                    #if sequence is broken, note it down in FIELD4
                    if frame_num - vid_track_info[id_num][FIELD3] > 1:
                        vid_track_info[id_num][FIELD4] += 1
                    #update info for this id
                    vid_track_info[id_num][FIELD1] += 1
                    vid_track_info[id_num][FIELD3] = frame_num
                
        track_info[fn] = vid_track_info
    with open("track_info.json", 'w+') as f:
        print("Writing to JSON file")
        json.dump(track_info, f, indent=4)
    return track_info

                        


    
    


if __name__ == "__main__":
    input_vid_dir = os.path.join(TEST_PATH, 'input_videos')
    output_dir = os.path.join(TEST_PATH, 'output')
    # print("Running deepsort, store output in output folder")
    # print(track(input_vid_dir, output_dir))

    bbox_dir = os.path.join(output_dir, 'bbox_output')
    print(get_track_info(bbox_dir))

