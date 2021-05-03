# -*- coding:utf-8 -*-
"""
This code is for capturing the appearance samples of your target patient.

"""


#%% 
import sys 
import os 
import os.path as osp 
from pathlib import Path 
from glob import glob


import cv2 
import numpy as np 
from PIL import Image  # for TensorFlow & PyTorch
import yaml  # (ref) https://pyyaml.org/wiki/PyYAMLDocumentation
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import tensorflow as tf 
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import core.utils as utils
from core.yolov4 import filter_boxes
from yolo_utils import (YOLO_INFER, 
                        CFG_FLAGS,

                        )


""" Path checking 
"""
python_ver = sys.version
script_path = os.path.abspath(__file__)
cwd = os.getcwd()
os.chdir(cwd) #changing working directory 

print(f"Python version: {Back.GREEN}{python_ver}{Style.RESET_ALL}")
print(f"The path of the running script: {Back.MAGENTA}{script_path}{Style.RESET_ALL}")
print(f"CWD is changed to: {Back.RED}{cwd}{Style.RESET_ALL}")


""" Load configurations in YAML
    (ref) https://hakurei.tistory.com/224
    (ref) https://zetcode.com/python/yaml/
    (ref) https://wikidocs.net/26  
"""
try: 
    with open('cfgs/capture_cfg.yaml', 'r') as cfg_yaml: 
        cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
        print("YAML is loaded o_< chu~")
        
except: 
    sys.exit("fail to load YAML...")
    

FLAGS = CFG_FLAGS(cfg)
print(vars(FLAGS)) #(ref) https://stackoverflow.com/questions/3992803/print-all-variables-in-a-class-python

GALLERY_DIR = cfg['CAPTURE_OPTIONS']['dataPath']['GALLERY_IMG_DIR']
CAPTURE_NUM = cfg['CAPTURE_OPTIONS']['NUM_IMG']
TARGET_NAME = cfg['CAPTURE_OPTIONS']['FILE_NAME']
CAM_NUM = cfg['CAMERA']['NUM']




# ================================================================= #
#                         1. Set device                             #
# ================================================================= #
#%% set the process device 
physical_devices = tf.config.list_physical_devices('GPU')  

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

else: 
    print(f"Device: {Back.YELLOW}processing on CPU{Style.RESET_ALL}")



def check_existdir(path):
    file_list = os.listdir(path)

    if len(file_list): 
        file_list.sort()   #(ref) https://wikidocs.net/16041
        frame_num = file_list[-1].split('.')[0].split('_')[-1]
        print(f'{Back.RED}Previous captured frame number:{Style.RESET_ALL} {frame_num}')
        
        return int(frame_num) + 1


    print(f"{Back.GREEN}No previous works{Style.RESET_ALL}")
    return 0 




def run(yolo_module, vid):
    
    path = f'{GALLERY_DIR}/{TARGET_NAME}'
    frame_id = check_existdir(path)   # return starting frame number 
    print(f"Start frame ID at :{frame_id }")    
    frame_count = 0 

    while True:
        return_value, frame  = vid.read()


        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)


        else: 
            raise ValueError("No image! Try with another video format")

        
        """ Inference 
        """
        bbox_list = yolo_module(frame)
#        print(f"num of bboxes: {len(bbox_list)}")

        if len(bbox_list):  # if not empty bbox 
            
            for bbox in bbox_list:
            
                x1, y1, x2, y2 = bbox

                RoI = image.crop((x1, y1, x2, y2))
                RoI.save(f'{GALLERY_DIR}/{TARGET_NAME}/{TARGET_NAME}_{frame_id:04}.jpg') #(ref) https://brownbears.tistory.com/483


                frame_count += 1
                frame_id += 1 
                print(f"{frame_count}/{CAPTURE_NUM}")

                
                if frame_count >= CAPTURE_NUM: 
                    
                    vid.release()
                    cv2.destroyAllWindows()                    
                    sys.exit("Storing is done... (+ _ +)v")

                

        if cv2.waitKey(1) & 0xFF == ord('q'): break  

    vid.release()
    cv2.destroyAllWindows()


#%%
if __name__ == "__main__":

    """ Make a directory for gallery 
    """
    DATA_DIR = Path(osp.join(GALLERY_DIR, TARGET_NAME ))
    DATA_DIR.mkdir(parents=True, exist_ok=True)


    """ init. the detector-tracker module 
    """
    yolo_module = YOLO_INFER(ConfigProto, InteractiveSession, FLAGS)


    """ init. camera object 
    """
    vid = cv2.VideoCapture(CAM_NUM)   # (ref) https://076923.github.io/posts/Python-opencv-2/
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    """ Run!!!
    """ 
    run(yolo_module, vid)







    