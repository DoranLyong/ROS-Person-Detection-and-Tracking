#!/usr/bin/env python3

import sys
import os
import os.path as osp
import random 

import cv2
import numpy as np
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from emergency_detection_management.srv import EmergencyDetec, EmergencyDetecRequest



""" Path checking 
"""
python_ver = sys.version
script_path = os.path.abspath(__file__)
cwd = osp.dirname(script_path) # get dir_name from the script path (ref) https://itmining.tistory.com/122
os.chdir(cwd) #changing working directory 

print(f"Python version: {Back.GREEN}{python_ver}{Style.RESET_ALL}")
print(f"The path of the running script: {Back.MAGENTA}{script_path}{Style.RESET_ALL}")
print(f"CWD is changed to: {Back.RED}{cwd}{Style.RESET_ALL}")



bridge = CvBridge()

def detect_emergency(imgs, task):




    if not imgs:
        sys.exit("no images")
        

    """ Assembling data 
    """

    my_msg = EmergencyDetecRequest()
    print("image bus length: ", len(my_msg.imgs))

    bbox =[] 

    for idx, img in enumerate(imgs):

        if len(my_msg.imgs) == idx: 
            print("Maximum batch")
            break

        my_msg.imgs[idx] = bridge.cv2_to_imgmsg(np.array(img), "bgr8")
        
        coordi = [0, 1,  2 , 3]     # example bbox: x1, y1, x2, y2 - order 
        bbox.extend(coordi)         # (ref) https://answers.ros.org/question/325559/how-can-we-actually-use-float32multiarray-to-publish-2d-array-using-python/

        
    print("Test bbox sequence: ", bbox)  # (ref) https://stackoverflow.com/questions/31369934/ros-publishing-array-from-python-node

    my_msg.bboxes = bbox
    print(my_msg.bboxes)
        


    """ Request & Response 
    """ 
    rospy.wait_for_service('fall_detection')

    try: 
        fall_detect = rospy.ServiceProxy('fall_detection', EmergencyDetec)

        print(f"Requesting...")

        # service call 
        res = fall_detect.call(my_msg)

        # service Response
        print("Responsed:")
        print("Task is : ", res.task)
        print("State is : ", res.state)


    except rospy.ServiceException as e:
        print("Service call failed: ", e)        






if __name__ == "__main__":

    cv_image1 = cv2.imread('1.png')
    cv_image2 = cv2.imread('2.png')
    

    samples = [cv_image1, cv_image2]

    imgs = [] 

    for idx, img in enumerate(samples):
        imgs.append(img)

    
    detect_emergency(imgs, task="fall_detection")

