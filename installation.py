

#Install Deepsort
import os
from os.path import exists, join, basename

project_name = "deep_sort_pytorch"
if not exists(project_name):
  # clone and install
  get_ipython().system('git clone -q --recursive https://github.com/ZQPei/deep_sort_pytorch.git')
  
import sys
sys.path.append(project_name)



# In[ ]:


#Download pretrained weights
yolo_pretrained_weight_dir = join(project_name, 'detector/YOLOv3/weight/')
if not exists(join(yolo_pretrained_weight_dir, 'yolov3.weights')):
  get_ipython().system('cd {yolo_pretrained_weight_dir} && wget -q https://pjreddie.com/media/files/yolov3.weights')

checkpoint_dir = os.path.join("deep_sort", "deep", "checkpoint")
deepsort_pretrained_weight_dir = join(project_name, checkpoint_dir)
if not exists(join(deepsort_pretrained_weight_dir, 'ckpt.t7')):
  file_id = '1_qwTWdzT9dWNudpusgKavj_4elGgbkUN'
  get_ipython().system('cd {deepsort_pretrained_weight_dir} && curl -Lb ./cookie "https://drive.google.com/uc?export=download&id={file_id}" -o ckpt.t7')

new_yolov3_path = os.path.join(project_name, "yolov3_deepsort.py")
if exists("yolov3_deepsort.py"):
  get_ipython().system('cp yolov3_deepsort.py {new_yolov3_path}')


