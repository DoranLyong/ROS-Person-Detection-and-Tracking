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
with open('capture_cfg.yaml', 'r') as cfg_yaml: 
    cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
    

GALLERY_DIR = cfg['dataPath']['GALLERY_IMG_DIR']
CAPTURE_NUM = cfg['OPTIONS']['NUM_IMG']
TARGET_NAME = cfg['OPTIONS']['FILE_NAME']




#%%






#%%
if __name__ == "__main__":

    """ Make a directory for gallery 
    """
    DATA_DIR = Path(GALLERY_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(TARGET_NAME)