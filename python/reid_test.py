# -*- coding:utf-8 -*-
"""
This code is for enrolling the appearance descriptors for gallery items
"""
#%% 
import sys 
import os 
import os.path as osp 

from tqdm import tqdm
import re  # regular expression (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/20/regex-usage-01-basic/
from PIL import Image
import cv2
import numpy as np 
import yaml 
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import torch 
import torchvision.transforms as T

from ReID_data import make_dataloader
from reid_utils import (ReID_FLAGS, 
                        ReID_INFERENCE,
                        euclidean_distance, 
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
def get_label(rank_list):
    """ get label info from Rank-10 results 
    """
    pattern = re.compile(r'([\d]+)_c([\S]+)_t')     # Regular expression ; (ref) https://wikidocs.net/4308
                                                    #                      (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate

    labels = [  pattern.search(item).groups()[-1] for item in rank_list ] # map(); (ref) https://dojang.io/mod/page/view.php?id=2286

    return labels 


#%% 
if __name__ == '__main__' :

    """ Set your device 
    """
    gpu_no = 0  # gpu_number 
    DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    print(f"device: { DEVICE }")    

    """ init. inference object  
    """
    model = ReID_INFERENCE(reid_flags, DEVICE)


    """ set dataloader 
    """
    transform = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    



    # sample query 
    query_path = 'reid_query_test.png'
    query_img = Image.open(query_path) 
#    query_img.show()

    input = torch.unsqueeze(transform(query_img), 0)  # [3, H, W] -> [1, 3, H, W] for torch tensor 
    input = input.to(DEVICE)    


    # loade gallery 

    gallery_feats = torch.load(osp.join(reid_flags.LOG_DIR, 'gfeats.pth')).to(DEVICE) # gallery features 
    gallery_img_path = np.load(osp.join(reid_flags.LOG_DIR, 'gallery_path.npy'))



    """ model inference 
    """ 
    query_feat = model(input)  # get feature in (1, 2048) tensor shape 
    norm_query = torch.nn.functional.normalize(query_feat, dim=1, p=2)  
#    print(f"check if normalized : {(norm_query**2).sum()} ")    # (ref) https://discuss.pytorch.org/t/question-about-functional-normalize-and-torch-norm/27755    



    """ feature metric 
    """
    dist_mat = euclidean_distance(norm_query, gallery_feats) # not bad & fast 
    indices = np.argsort(dist_mat, axis=1)  # get index order in the best order (short distnace first)

    rank_list = gallery_img_path[indices[0, :5]]  # control Rank number 
    print(f"Finding ID of {query_path}")
    print(f"Rank-{len(rank_list)} : {rank_list}")  # Rank-5 results    


    labels = get_label(rank_list)
    print(f"Labels: {labels}", end='\n')

    # get most frequent element
    res = max(set(labels), key = labels.count)  # (ref) https://www.geeksforgeeks.org/python-element-with-largest-frequency-in-list/
    ratio = labels.count(res) / len(labels)     # (ref) https://vision-ai.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A6%AC%EC%8A%A4%ED%8A%B8-count-%EC%99%80-len

    print(f"Query is {Back.GREEN}{res}{Style.RESET_ALL}")
    print(f"'{res}' occupies {Back.GREEN}{ratio*100}%{Style.RESET_ALL} among Rank-{len(rank_list)}")



    """ visualize 
    """
    visualizer(query_img, gallery_img_path , query_path, indices, camid='mixed', top_k = len(rank_list), img_size=[256,128])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
