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
def get_label(rank_list):
    """ get label info from Rank-10 results 
    """
    pattern = re.compile(r'([\d]+)_c([\S]+)_t')     # Regular expression ; (ref) https://wikidocs.net/4308
                                                    #                      (ref) https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate

    labels = [  pattern.search(item).groups()[-1] for item in rank_list ] # map(); (ref) https://dojang.io/mod/page/view.php?id=2286

    return labels 



#%% 
# ================================================================= #
#                             ROS services                          #
# ================================================================= #
""" (ref) https://stackoverflow.com/questions/62938146/getting-realsense-depth-frame-in-ros
"""



class PersonDetector(object):

    cv2_img_buffer = list()  # class variable ; 
                             # (ref) http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-oop-part-3-%ED%81%B4%EB%9E%98%EC%8A%A4-%EB%B3%80%EC%88%98class-variable/
    
    bridge = CvBridge()

    def __init__(self, topic):
        super().__init__()  # (ref) https://dojang.io/mod/page/view.php?id=2386
        print("detector init.")

        """ to get images 
        """

        self.cv2_img = None
        
        self.topic = topic
        

        rospy.Subscriber(self.topic , Image , PersonDetector.imageCallback)
        


    @classmethod    
    def imageCallback(cls, img_msg):  # @classmethod; (ref) https://www.geeksforgeeks.org/class-method-vs-static-method-python/
        
        try:
            cv2_img  = PersonDetector.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')   # Cvt ROS image msg to cv2 img 
                                                                                                # (ref) http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
#            print("image callback: ", PersonDetector.cv2_img )
            
#            cv2.imshow("test", PersonDetector.cv2_img )
#            cv2.waitKey(32)

            PersonDetector.cv2_img_buffer.append(cv2_img)

            if len(PersonDetector.cv2_img_buffer) == img_buffer + 1:
                PersonDetector.cv2_img_buffer.pop(0)   # keep the image buffer size in 10.

        except CvBridgeError as e:
            print(e)
            return


        
    def __call__(self, yolo_module, reid_model):
        

        print("Yolo inference...")

        """ load ReID gallery
        """
        gallery_feats = torch.load(osp.join(reid_flags.LOG_DIR, 'gfeats.pth')).to(DEVICE) # gallery features 
        gallery_img_path = np.load(osp.join(reid_flags.LOG_DIR, 'gallery_path.npy'))


        imgs = [] 
        bboxes = [] 
        
        while True:

            if len(PersonDetector.cv2_img_buffer) == img_buffer:
                """ if you get images 
                """

                for idx, img in enumerate(PersonDetector.cv2_img_buffer):
                    
                    title = "./check_results/captured/" + str(idx) + "_test.png"
                    cv2.imwrite(title, img)


                    """ Frame preprocess 
                    """
                    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame_copy = frame.copy()
                    image = pil_Image.fromarray(frame)

            

                    """ Yolo & DeepSORT Inference             
                    """            
                    bbox_list, id_list  = yolo_module(frame)
                    
                    if len(bbox_list) == 0:
                        """ No detection
                        """
                        continue
                    
                    print("num of bboxes: ", len(bbox_list))



                    detect_info = {}
                    if len(bbox_list):  # if not empty bbox 
                        bbox_np = np.asarray(bbox_list) # (ref) https://supermemi.tistory.com/66
                        id_np = np.asarray(id_list).reshape(len(id_list) ,-1)

                        id_bboxes = np.concatenate((id_np, bbox_np), axis=1)  # (ref) https://supermemi.tistory.com/66
#                        print(f"id_bboxes: {id_bboxes}")     



                        for ind, id_bbox in enumerate(id_bboxes):
                            id = id_bbox[0]
                            x1, y1, x2, y2 = id_bbox[1:]

                            RoI = image.crop((x1, y1, x2, y2))
                            query_img = RoI.resize((64, 128))  # (ref) https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
                                                                # for ReID model infernce; (ref) https://github.com/DoranLyong/person-reid-tiny-baseline

                            title = "./check_results/query/" + str(ind + idx) +  "_ID_" + str(id) + ".png"
                            cv2.imwrite(title, np.asarray(query_img) )                                                                



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

#                            print("labels: ", labels)      


                            # get most frequent element
                            res = max(set(labels), key = labels.count)  # get person name (ref) https://www.geeksforgeeks.org/python-element-with-largest-frequency-in-list/
                            ratio = labels.count(res) / len(labels)      # (ref) https://vision-ai.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A6%AC%EC%8A%A4%ED%8A%B8-count-%EC%99%80-len              


                            print("Query ID ", id ," is ", res)
                            print(res,  "occupies" ,ratio*100 , "'%' ") 


                            if ratio > 0.6:  # > 60% among Rank-5
                                """ Reasonable ReID result 
                                """
                                detect_info[res] = (x1, y1, x2, y2)  # {'cls_name' : bbox_coordi}


                    if len(detect_info) :
#                        print(detect_info)    
                        target_bbox = [bbox for target, bbox in detect_info.items() if target in ('PATIENT')]  # get the only bbox of the target 'PATIENT' class    

                        target_bbox = list(itertools.chain.from_iterable(target_bbox)) # unpack tuple in list 

#                        print("target_bbox: ", target_bbox) # x1, y1, x2, y2 order 

                        imgs.append(img)
                        bboxes.append(target_bbox)

                return imgs, bboxes

            
            








#%% 
# ================================================================= #
#                             ROS services                          #
# ================================================================= #

def emergency_detection(percep_req):
    """ Receive call from perception_management 
    """ 
    print("Received task requisition: ", percep_req.task)




    """ Get image sequence with Yolo + DeepSORT 
    """ 
    img_msg = EmergencyDetecRequest()





    # 여기 다음주에 와서 YoLO+DeepSORT붙이기 
    # realsense 영상으로 topic subscribe 

    

    yolo_module = YOLO_INFER(ConfigProto, InteractiveSession, FLAGS) # init. the detector-tracker module 
    reid_model = ReID_INFERENCE(reid_flags, DEVICE)


    detector = PersonDetector(topic=camera_topic) # class init. and call 
    imgs, bboxes = detector(yolo_module, reid_model)  # get two lists for images and bboxes 

    

    if len(imgs): 
        print("############ Target_seq is detected ###############")
        
        bbox_seq = []

        for idx, img in enumerate(imgs): 

            

            x1, y1, x2, y2  = bboxes[idx]
            img = cv2.rectangle(img, (x1, y1) , (x2, y2), (0, 255,0), 3 )

            title = "./check_results/bbox/" + str(idx) + ".png"
            cv2.imwrite(title, img)


            bbox_seq.extend(bboxes[idx]) 
            img_msg.imgs.append(bridge.cv2_to_imgmsg(np.array(img), "bgr8"))  


#        print(bboxes)
#        print(bbox_seq)

        img_msg.bboxes = bbox_seq

    else: 
        print("############ No target detected ###############")





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
    