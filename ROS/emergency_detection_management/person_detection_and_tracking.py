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


import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image  # for TensorFlow & PyTorch
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
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from emergency_detection_management.srv import EmergencyDetec, EmergencyDetecRequest
from perception_msgs.srv import persontracking, persontrackingResponse




# ================================================================= #
#                         ROS configurations                        #
# ================================================================= #
bridge = CvBridge()




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


#%% Load configurations in YAML for ReID
try: 
    with open('./cfgs/reid_cfg.yaml', 'r') as cfg_yaml: 
        cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
        print("YAML for ReID is loaded o_< chu~")
        
except: 
    sys.exit("fail to load YAML for ReID...")


reid_flags = ReID_FLAGS(cfg)
#print(vars(reid_flags))  # check the class members(ref) https://www.programiz.com/python-programming/methods/built-in/vars



#%% 
# ================================================================= #
#                         1. Set device                             #
# ================================================================= #
#%% set the process device for TensorFlow2 
physical_devices = tf.config.list_physical_devices('GPU')  

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

else: 
    print("Device for TF2: processing on CPU")


#%% Set your device for PyTorch 

if torch.cuda.is_available(): 
    DEVICE = torch.device('cuda:0')

else: 
    DEVICE = torch.device('cpu')

#DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
print("Device for PyTorch: ",  DEVICE )      




#%% 
# ================================================================= #
#                      Set transforms for ReID                      #
# ================================================================= #
#%% Image transformation for ReID 
transform = T.Compose([
                        T.Resize([256, 128]),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])    


#%%
Rank_num = 5  # Rank-5,  Rank-10





#%% 
# ================================================================= #
#                             ROS services                          #
# ================================================================= #



def task_switch(value:str): 
    return {
        "fall_detection" : "fall_detection_func",
        "goingout_detection" : "goingout_func"
        
    }.get(value, -1) # (ref) https://inma.tistory.com/141





def emergency_detection(percep_req):
    """ Receive call from perception_management 
    """ 
    print("Received task requisition: ", percep_req.task)










    """ Get image sequence with Yolo + DeepSORT 
    """ 
    img_msg = EmergencyDetecRequest()

    cv_image1 = cv2.imread('1.png')
    cv_image2 = cv2.imread('2.png')    


    img_msg.imgs[0] = bridge.cv2_to_imgmsg(np.array(cv_image1), "bgr8")
    img_msg.imgs[1] = bridge.cv2_to_imgmsg(np.array(cv_image2), "bgr8")

    bbox =[] 
    coordi = [1, 2, 3, 4, 5, 6, 7, 8]
    bbox.extend(coordi)  


    img_msg.bboxes = bbox



    # 여기 다음주에 와서 YoLO+DeepSORT붙이기 
    # realsense 영상으로 topic subscribe 

















    """ Request to :
                    {'fall_detection', 'goingout_detection'}
    """
    print("Request task is : ", percep_req.task)

    rospy.wait_for_service(percep_req.task)

    try: 
        adnormal_detector =  rospy.ServiceProxy(percep_req.task, EmergencyDetec)

        print(f"Requesting...")


        # service call 
        res = adnormal_detector.call(img_msg)



        # service Response
        print("Responsed:")
        print("From: ", res.task)
        print("State is : ", res.state)


    except rospy.ServiceException as e:
        print("Service call failed: ", e)       



    """ Response to perception_management
    """
    percep_res = persontrackingResponse()

    percep_res.state = res.state

    return percep_res









#%% 
if __name__ == "__main__":

    Node_name = 'person_detect_tracking_srv'
    Bus_name = 'person_detect_tracking_srv'

    rospy.init_node( Node_name )

    s = rospy.Service(Bus_name, persontracking, emergency_detection)

    

    try: 
        rospy.spin() 

    except KeyboardInterrupt: 
        print("Shutting down...")     
    