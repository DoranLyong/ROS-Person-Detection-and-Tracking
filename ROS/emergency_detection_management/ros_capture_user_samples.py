#!/usr/bin/env python3

import sys 
import os 
import os.path as osp 
from pathlib import Path 
from glob import glob
import itertools
import re  # regular expression (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/


""" Path checking 
"""
python_ver = sys.version
script_path = os.path.abspath(__file__)
cwd = osp.dirname(script_path) # get dir_name from the script path (ref) https://itmining.tistory.com/122
os.chdir(cwd) #changing working directory 

print("Python version: ", python_ver)
print("The path of the running script: ", script_path)
print("CWD is changed to: ", cwd)
print("python3 executable path: ", sys.executable)
print("python3 lib path: ", sys.path)


import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image as pil_Image  # for TensorFlow & PyTorch
import yaml  # (ref) https://pyyaml.org/wiki/PyYAMLDocumentation
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import tensorflow as tf 
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import torch 
import torchvision.transforms as T

import core.utils as utils
from core.yolov4 import filter_boxes
from yolo_utils import (YOLO_INFER, 
                        CFG_FLAGS,
                        )
from ReID_data import make_dataloader
from reid_utils import (ReID_FLAGS, 
                        ReID_INFERENCE,
                        euclidean_distance, 
                        cosine_similarity, 
                        vis_tensorImg,
                        visualizer, 
                        )

import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


from emergency_detection_management.srv import EmergencyDetec, EmergencyDetecRequest
from perception_msgs.srv import persontracking, persontrackingResponse




# ================================================================= #
#                         ROS configurations                        #
# ================================================================= #
bridge = CvBridge()
camera_topic = '/camera/color/image_raw'  # check the depth image topic in your Gazebo environmemt and replace this with your
img_buffer = 30  # the size of image sequence to spend to the server node. 



# ================================================================= #
#                         Get configurations                        #
# ================================================================= #
#%% YAML for yolo & DeepSORT
""" Load configurations in YAML
    (ref) https://hakurei.tistory.com/224
    (ref) https://zetcode.com/python/yaml/
    (ref) https://wikidocs.net/26  
"""
try: 
    with open('cfgs/capture_cfg.yaml', 'r') as cfg_yaml: 
        cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
        print("YAML for yolo & DeepSORT is loaded o_< chu~")
        
except: 
    sys.exit("fail to load YAML for yolo & DeepSORT...")
    

FLAGS = CFG_FLAGS(cfg)
#print(vars(FLAGS)) #(ref) https://stackoverflow.com/questions/3992803/print-all-variables-in-a-class-python                        

CAM_NUM = cfg['CAMERA']['NUM']  # webcam device number 




    
