# -*- coding:utf-8 -*-
"""
This code is for enrolling the appearance descriptors for gallery items
"""
#%% 
import sys 
import os 
import os.path as osp 

from tqdm import tqdm
import numpy as np 
import yaml 
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import torch 
import torchvision.transforms as T

from ReID_data import make_dataloader
from reid_utils import (ReID_FLAGS, 
                        ReID_INFERENCE,
                        vis_tensorImg
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


#%% Load configurations in YAML
try: 
    with open('./cfgs/reid_cfg.yaml', 'r') as cfg_yaml: 
        cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
        print("YAML is loaded o_< chu~")
        
except: 
    sys.exit("fail to load YAML...")


reid_flags = ReID_FLAGS(cfg)
#print(vars(reid_flags))  # check the class members(ref) https://www.programiz.com/python-programming/methods/built-in/vars





#%% 
if __name__ == '__main__':

    """ Set your device 
    """
    gpu_no = 0  # gpu_number 
    DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    print(f"device: { DEVICE }")    

    """ init. inference object  
    """
    model = ReID_INFERENCE(reid_flags, DEVICE)


    """ dataloader for gallery 
    """
    gallery_loader, num_classes = make_dataloader(reid_flags)  # (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/datasets

    img_path_list = []
    feats, pids = [], []


    loop = tqdm(enumerate(gallery_loader), total=len(gallery_loader))  
    for n_iter, (img, pid, img_path) in loop:
        img = img.to(DEVICE)  # get one input data 
#        vis_tensorImg(img[0])  # for checking images  

        feat = model(img)  # get feature in (Batch_size, 2048) tensor shape 

        """ update list
        """ 
        img_path_list.extend(img_path)   # (ref) https://wikidocs.net/14#extend
                                        # extend vs. append (ref) https://www.edureka.co/community/5916/difference-between-append-vs-extend-list-methods-in-python  
        pids.extend(np.asarray(pid))
        feats.append(feat)  # (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/utils/metrics.py        




    """ Compute G_FEATS
        (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/utils/metrics.py
    """
    feats_tensor = torch.cat(feats, dim=0) # [N, 2048]
    norm_feats = torch.nn.functional.normalize(feats_tensor, dim=1, p=2)    # along channel ; (ref) https://pytorch.org/docs/master/generated/torch.nn.functional.normalize.html
                                                                            # p=2 for L2-norm 


    gallery_features = norm_feats
    


    """ Save metrics
        (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/processor/processor.py
    """
    print(f"Save gallery features")

    log_path = reid_flags.LOG_DIR
    np.save(osp.join(log_path , 'gallery_path.npy') , img_path_list) # gallery path

    torch.save(gallery_features, osp.join(log_path , 'gfeats.pth' ))

