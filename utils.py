#!/usr/bin/env python
# coding: utf-8

# Get every n-th frame of the video and save it to the specified directory

# In[ ]:


import cv2, os, pickle, subprocess, sys
#input file must be in default current directory
def extract_frames(input_fp, input_fn, output_folder):
  logger = create_logger()
  logger.info(f"Extracting {input_fp} to {output_folder}")
  cap = cv2.VideoCapture(input_fp)
  count = 0 #frame number starting from 1

  while cap.isOpened():
      ret, frame = cap.read()

      if ret:
          #Set grayscale colorspace for the frame. 
          # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          # cv2.imwrite(save_dir + '/output_frame{:d}.jpg'.format(count+1), gray)
          output_dir = output_folder + f'/{input_fn[:-4]}'
          if not os.path.exists(output_dir):
            # get_ipython().system('mkdir $output_dir')
            os.makedirs(output_dir)
          cv2.imwrite(output_dir + '/output_frame{:03d}.jpg'.format(count+1), frame)
          count += 1 # advancing frame by 1
          cap.set(1, count)
      else:
          cap.release()
          break
  if not count:
    logger.warning(f"{input_fp} has no video frame extracted!")


# Extract bounding box of input video file located in /frames and save each extracted box to a jpg file. Run this section in the order: run deepsort -> extract frame -> this section

# In[ ]:


#extract bounding box of input video file located in /frames 
#param input_fn: Input video filename to extract bounding box from
#param pickle_path: saved bbox.pkl file containing bounding box info of that video
#param input_frames_folder: the folder containing extracted frames of each video file. 
def extract_bbox(input_fn, pickle_path, input_frames_folder):
  logger = create_logger()
  logger.info(f"Extracting bounding box from: {input_fn}")
  with open(pickle_path, 'rb') as infile:
    saved_bbox_info = pickle.load(infile)

#   BASE_FRAME_DIR = os.getcwd() + input_frames_folder #base directory storing information of each frames
  base_dir = os.path.join(input_frames_folder, input_fn)

  for framenum_and_box in saved_bbox_info:
    frame_num, identities, bboxs = framenum_and_box

    # #create directory to store information for current frame
    # frame_dir = base_dir + '/frame_{:03d}'.format(frame_num)
    # if not os.path.exists(frame_dir):
    #   !mkdir $frame_dir

    #crop out image bounded by box
    frame_img_fp = os.path.join(base_dir,"output_frame{:03d}.jpg".format(frame_num))
    if not os.path.exists(frame_img_fp):
      logger.warning("File: ", frame_img_fp, " Not Found. Skipping...")
      continue
    frame_img = cv2.imread(frame_img_fp)

    for i, bbox in enumerate(bboxs):
      detected_id = identities[i]

      #create directory for specific id to store cropped frames of this detected id
      id_dir = os.path.join(base_dir, f'id_{detected_id:02d}')
      if not os.path.exists(id_dir):
        # get_ipython().system('cd $base_dir && mkdir $id_dir')
        os.makedirs(id_dir)
      
      x1,y1,x2,y2 = bbox
      crop_id_img = frame_img[y1:y2, x1:x2]
      save_fn = os.path.join(id_dir,  f"frame{frame_num:03d}_id{detected_id:02d}.jpg")

      cv2.imwrite(save_fn, crop_id_img)
      cv2.waitKey(0)


def run_deepsort(output_dir, input_vid_p, mobile, called_dir, base_dir):
  logger=  create_logger()
  deepsort_dir = os.path.join(base_dir, "deep_sort_pytorch")
  final_vid_dir = os.path.join(output_dir , 'final_vid')
  bbox_dir = os.path.join(output_dir, 'bbox_output')
  input_vid_dir = os.path.dirname(input_vid_p)
  input_vid_fn = os.path.basename(input_vid_p)
  logger.info(f"Changing working directory to: {deepsort_dir}")
  os.chdir(deepsort_dir)

  logger.info(f"Running Deepsort on {input_vid_p}. output to {output_dir}")
  logger.debug(f"Directory paths:\nDeepsort_dir: {deepsort_dir}, Final Video: {final_vid_dir}, Bbox Dir: {bbox_dir}, Input Vid Dir: {input_vid_dir}")

  if not os.path.exists(final_vid_dir):
    logger.info(f"Creating final video dir")
    os.makedirs(final_vid_dir)
  if not os.path.exists(bbox_dir):
    logger.info("Creating bbox dir")
    os.makedirs(bbox_dir)

  if mobile:
    #change metadata
    logger.info("Correcting mobile video metadata")
    temp_dir = os.path.join(input_vid_dir, 'my_temp')
    if not os.path.exists(temp_dir):
      os.makedirs(temp_dir)
    # subprocess.call(["ffmpeg", "-y", "-i", input_vid_p, "-metadata:s:v", "rotate='0'", "-c:v", input_vid_p])
    subprocess.call(["ffmpeg", "-y", "-i", input_vid_p, "-metadata:s:v", "rotate='0'", "-vf", "transpose=1", "-c:v", "libx264", "-crf", "23", "-acodec", "copy", os.path.join(temp_dir, "temp.mp4") ])
    subprocess.call(["ffmpeg", "-y", "-i",     os.path.join(temp_dir, "temp.mp4"), "-metadata:s:v", "rotate='0'", "-vf", "transpose=2", "-c:v", "libx264", "-crf", "23", "-acodec", "copy", input_vid_p ])

  #perform deepsort
  logger.info("Calling yolov3_deepsort.py in deepsort_pytorch")
  subprocess.run([sys.executable,   os.path.join(deepsort_dir, "yolov3_deepsort.py"), "--ignore_display", input_vid_p, '--save_path',os.path.join(final_vid_dir, "openpose.avi")])
  os.rename(  os.path.join(deepsort_dir, "bounding_box.pkl"),   os.path.join(bbox_dir, f"{input_vid_fn[:-4]}_bbox.pkl"))
  # convert the result into MP4
  subprocess.call(["ffmpeg", "-y", "-loglevel", "info", "-i",   os.path.join(final_vid_dir, "openpose.avi"),   os.path.join(final_vid_dir, input_vid_fn)])
  os.chdir(called_dir) #change back to original directory
  logger.info(f"Changing back working directory to: {called_dir}")

  # In[ ]:

def get_f1(y_true, y_pred): #taken from old keras source code
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val


#This function is to create a logger which logs to file (at warning level) and console (at debug level)
import logging
def create_logger():
  logging.root.handlers = []
  logging.basicConfig(format='%(module)s : %(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG , filename='Output.log')

  # set up logging to console
  console = logging.StreamHandler()
  console.setLevel(logging.DEBUG)
  # set a format which is simpler for console use
  formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
  console.setFormatter(formatter)
  logging.getLogger("").addHandler(console)
  logger = logging.getLogger(__name__)
  return logger

