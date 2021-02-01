#!/usr/bin/env python
# coding: utf-8

# # Run Prediction

# ### 1. Take in input video (Configure here)

# In[1]:
from pathlib import Path
import os,sys, tensorflow as tf, numpy as np
from os.path import exists, join, basename
from tensorflow.keras import backend as K
import subprocess

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
from prepare_img_data import load_predict_data_vid, normalize_img_data
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
logger = create_logger()

PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction')
#set image width and height for prediction. Must be same with the trained model shape
IMG_HEIGHT, IMG_WIDTH = 700,100
#set classification model name
MODEL_NAME = 'seq30_128dense_lost0.58_f1-0.7_100x700.h5'


def run_prediction(video_fp, mobile, output_dir, custom_metric=False, debug=False):
  logger.info("Running prediction")
  input_vid_fn = os.path.basename(video_fp)
  input_vid_dir = os.path.dirname(video_fp)


  logger.info(f"Variables set --- Model: {MODEL_NAME}, Img_size: {IMG_WIDTH}x{IMG_HEIGHT}")

  # ### 2. Run DeepSORT on input video
  logger.info(f"Running Deepsort on {video_fp}. output to {output_dir}")
  run_deepsort(output_dir, video_fp, mobile, called_dir, BASE_DIR)

  # ### 3. Extract Frames to readable input for model.
  #Extract frames to frames folder
  frames_folder = os.path.join(output_dir, 'frames')
  if not os.path.exists(frames_folder):
      logger.info("Creating frames folder: ", frames_folder)
      os.makedirs(frames_folder)
  extract_frames(video_fp, input_vid_fn, frames_folder)


  # In[8]:
  #Extract bbox of the frames
  bbox_dir = os.path.join(output_dir, 'bbox_output')
  pickle_path = os.path.join(bbox_dir, f"{input_vid_fn[:-4]}_bbox.pkl")
  extract_bbox(input_vid_fn[:-4], pickle_path, frames_folder)


  # ### 4. Read in the trained model
  logger.info(f"Loading model: {MODEL_NAME}")
  model_fp = os.path.join(PREDICTION_DIR, "model", MODEL_NAME)

  # In[9]:

  if os.path.exists(model_fp):
      if custom_metric:
          model = tf.keras.models.load_model(
          model_fp, custom_objects={"get_f1":get_f1}, compile=True
      )
      else:
          model = tf.keras.models.load_model(
          model_fp, custom_objects=None, compile=True
      )
  else:
      logger.error("Missing model for prediction")
      raise Exception("Missing model")

  # ### 5. Make prediction

  # In[10]:

  logger.info("Make prediction")
  vid_frames_folder = os.path.join(frames_folder, input_vid_fn[:-4])
  my_x, names, images= load_predict_data_vid(vid_frames_folder, input_vid_fn[:-4], debug=debug)
  my_x = normalize_img_data(my_x, IMG_HEIGHT, IMG_WIDTH)
  my_x = np.array(my_x)
  logger.info(f"Input shape is: {my_x.shape}")
  res = []
  for i in range(len(my_x)):
      val = model.predict(np.array([my_x[i,]]))[0]
      # y_pred.append(0 if val <= 0.5 else 1)
      print(f"Predicted prob: {val}\tname: {names[i]}")
      res.append({
        "ID": names[i],
        "filepath": images[i],
        "mood": ("Depressed" if val < 0.5 else "Healthy"),
        "conf": str(val[0])
      })
  return res


  # In[6]:




if __name__ == "__main__":
  input_vid_name = "jing.mp4"
  logger.info(f"Input video filename: {input_vid_name}")

  output_dir = os.path.join(PREDICTION_DIR, 'output')
  logger.info(f"Processed frames directory: {output_dir}")

  fp = os.path.join(PREDICTION_DIR, "input_vid", input_vid_name)
  print("~~~~~~~~~~~~~~~~~Predicted results~~~~~~~~~~~`")
  print(run_prediction(fp, True, output_dir, custom_metric=True))
       