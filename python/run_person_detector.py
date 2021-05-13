# -*- coding:utf-8 -*-
"""
This code is the is main for person detection with person re-identification module to search the target.
Before running this code, please complete the following processes:
    - capturing user samples 
    - enrolling person appearance features & paths
"""

#%% 
import sys 
import os 
import os.path as osp 
from pathlib import Path 
from glob import glob
import itertools
import re  # regular expression (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/

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



""" Path checking 
"""
python_ver = sys.version
script_path = os.path.abspath(__file__)
cwd = os.getcwd()
os.chdir(cwd) #changing working directory 

print(f"Python version: {Back.GREEN}{python_ver}{Style.RESET_ALL}")
print(f"The path of the running script: {Back.MAGENTA}{script_path}{Style.RESET_ALL}")
print(f"CWD is changed to: {Back.RED}{cwd}{Style.RESET_ALL}")





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
        print(f"YAML for {Back.YELLOW}yolo & DeepSORT{Style.RESET_ALL} is loaded o_< chu~")
        
except: 
    sys.exit("fail to load YAML for yolo & DeepSORT...")
    

FLAGS = CFG_FLAGS(cfg)
#print(vars(FLAGS)) #(ref) https://stackoverflow.com/questions/3992803/print-all-variables-in-a-class-python                        

CAM_NUM = cfg['CAMERA']['NUM']  # webcam device number 


#%% Load configurations in YAML for ReID
try: 
    with open('./cfgs/reid_cfg.yaml', 'r') as cfg_yaml: 
        cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
        print(f"YAML for {Back.YELLOW}ReID{Style.RESET_ALL} is loaded o_< chu~")
        
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
    print(f"Device for TF2: {Back.YELLOW}processing on CPU{Style.RESET_ALL}")


#%% Set your device for PyTorch 
gpu_no = 0  # gpu_number 
DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
print(f"Device for PyTorch: {Back.YELLOW}{ DEVICE }{Style.RESET_ALL}")      




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
def get_label(rank_list):
    """ get label info from Rank-10 results 
    """
    pattern = re.compile(r'([\d]+)_c([\S]+)_t')     # Regular expression ; (ref) https://wikidocs.net/4308
                                                    #                      (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate

    labels = [  pattern.search(item).groups()[-1] for item in rank_list ] # map(); (ref) https://dojang.io/mod/page/view.php?id=2286

    return labels 



#%%
def vis_resutls(frame , detect_info:dict):
    img = frame.copy()

    #initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
    for idx, (person_name, bbox) in enumerate(detect_info.items()):  # (ref) https://stackoverflow.com/questions/36244380/enumerate-for-dictionary-in-python
    
        """ Draw bbox on screen 
        """    
        color = colors[int(idx) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(person_name)+len(str(idx)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, person_name ,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)


    result = np.asarray(img)
    result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)         

    cv2.imshow("ReId results", result)   

#%% 
def run(yolo_module, vid):

    """ init. ReId inference object  
    """    
    
    reid_model = ReID_INFERENCE(reid_flags, DEVICE)

    """ load ReID gallery
    """
    gallery_feats = torch.load(osp.join(reid_flags.LOG_DIR, 'gfeats.pth')).to(DEVICE) # gallery features 
    gallery_img_path = np.load(osp.join(reid_flags.LOG_DIR, 'gallery_path.npy'))




    while True:
        return_value, frame  = vid.read()


        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_copy = frame.copy()
            image = Image.fromarray(frame)


        else: 
            raise ValueError("No image! Try with another video format")

        
        """ Yolo & DeepSORT Inference 
        """
        bbox_list, id_list = yolo_module(frame)
#        print(f"num of bboxes: {len(bbox_list)}")



        detect_info = {}
        if len(bbox_list):  # if not empty bbox 
            bbox_np = np.asarray(bbox_list) # (ref) https://supermemi.tistory.com/66
            id_np = np.asarray(id_list).reshape(len(id_list) ,-1)

            id_bboxes = np.concatenate((id_np, bbox_np), axis=1)  # (ref) https://supermemi.tistory.com/66
#            print(f"id_bboxes: {id_bboxes}")
    
            
            for id_bbox in id_bboxes:
            
                id = id_bbox[0]
                x1, y1, x2, y2 = id_bbox[1:]

                RoI = image.crop((x1, y1, x2, y2))
                query_img = RoI.resize((64, 128))  # (ref) https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
                                                    # for ReID model infernce; (ref) https://github.com/DoranLyong/person-reid-tiny-baseline

#                cv2.imshow(f"ID: {id}", np.asarray(query_img) )


                """ ReID inference  
                """
                query_input = torch.unsqueeze(transform(query_img), 0)  # [3, H, W] -> [1, 3, H, W] for torch tensor 
                query_input = query_input.to(DEVICE)    

                query_feat = reid_model(query_input)  # get feature in (1, 2048) tensor shape 
                norm_query = torch.nn.functional.normalize(query_feat, dim=1, p=2) # feature normalization



                """ feature metric for ReID 
                """
                dist_mat = euclidean_distance(norm_query, gallery_feats) # not bad & fast 
                
                indices = np.argsort(dist_mat, axis=1)  # get index order in the best order (short distnace first)

                rank_list = gallery_img_path[indices[0, :Rank_num]]  # control Rank number 
                labels = get_label(rank_list) 
                print(f"{labels}")

                # get most frequent element
                res = max(set(labels), key = labels.count)  # get person name (ref) https://www.geeksforgeeks.org/python-element-with-largest-frequency-in-list/
                ratio = labels.count(res) / len(labels)     # (ref) https://vision-ai.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A6%AC%EC%8A%A4%ED%8A%B8-count-%EC%99%80-len

                print(f"Query ID {id} is {Back.GREEN}{res}{Style.RESET_ALL}")
                print(f"'{res}' occupies {Back.GREEN}{ratio*100}%{Style.RESET_ALL} among Rank-{len(rank_list)}")                

                if ratio > 0.6:  # > 60% among Rank-5
                    """ Reasonable ReID result 
                    """
                    detect_info[res] = (x1, y1, x2, y2)  # {'cls_name' : bbox_coordi}
                
        
        if len(detect_info) :
            print(detect_info)
            vis_resutls( frame_copy, detect_info)

            target_bbox = [bbox for cls, bbox in detect_info.items() if cls in ('PATIENT')]  # get the only bbox of the target 'PATIENT' class
            target_bbox = list(itertools.chain.from_iterable(target_bbox)) # unpack tuple in list ;  (ref) https://stackoverflow.com/questions/13958998/python-list-comprehension-unpacking-and-multiple-operations

            print(f"target_bbox: {target_bbox}") # x1, y1, x2, y2 order 
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break  

    vid.release()
    cv2.destroyAllWindows()



#%% 
#%%
if __name__ == "__main__":

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
