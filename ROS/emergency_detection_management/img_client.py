#!/usr/bin/env python3

import sys
import os
import os.path as osp

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
    cv_image1 = cv2.imread('1.png')

    my_msg = EmergencyDetecRequest()

    for idx, img in enumerate(imgs):
        cv2.imshow("Before cv_bridge", img)
        cv2.waitKey(0)

        my_msg.imgs[idx] = bridge.cv2_to_imgmsg(np.array(img), "bgr8")



    """ Request 
    """ 
    rospy.wait_for_service('fall_detection')

    try: 
        pass 


    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")




    cv2.imshow("test", imgs[0])
    cv2.waitKey(0)






if __name__ == "__main__":

    cv_image1 = cv2.imread('1.png')
    cv_image2 = cv2.imread('2.png')

    samples = [cv_image1, cv_image2]

    imgs = [] 

    for idx, img in enumerate(samples):
        imgs.append(img)

    
    detect_emergency(imgs, task="fall_detection")

