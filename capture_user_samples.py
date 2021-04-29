# -*- coding:utf-8 -*-
"""
This code is for capturing the appearance samples of your target patient.

"""


#%% 
import sys 
import os 
import os.path as osp 
from pathlib import Path 


import cv2 
import numpy as np 
import yaml  # (ref) https://pyyaml.org/wiki/PyYAMLDocumentation
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import tensorflow as tf 
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import core.utils as utils
from core.yolov4 import filter_boxes
from yolo_utils import (YOLO_INFER, 
                        YOLO_CFG,

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
    

FLAGS = YOLO_CFG(cfg)
print(vars(FLAGS)) #(ref) https://stackoverflow.com/questions/3992803/print-all-variables-in-a-class-python

GALLERY_DIR = cfg['dataPath']['GALLERY_IMG_DIR']
CAPTURE_NUM = cfg['OPTIONS']['NUM_IMG']
TARGET_NAME = cfg['OPTIONS']['FILE_NAME']




# ================================================================= #
#                         1. Set device                             #
# ================================================================= #
#%% set the process device 
physical_devices = tf.config.list_physical_devices('GPU')  

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

else: 
    print(f"Device: {Back.YELLOW}processing on CPU{Style.RESET_ALL}")







#%%
if __name__ == "__main__":

    """ Make a directory for gallery 
    """
    DATA_DIR = Path(GALLERY_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


    YOLO_INFER(ConfigProto, InteractiveSession, FLAGS)


    