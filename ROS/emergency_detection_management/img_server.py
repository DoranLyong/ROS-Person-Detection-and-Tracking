#!/usr/bin/env python3

import sys
import os
import os.path as osp

import cv2
import numpy as np
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from emergency_detection_management.srv import EmergencyDetec, EmergencyDetecResponse



bridge = CvBridge()


def fall_detection(req):

    """ Receive call 
    """ 
    print(f"")
    print(f"Received data length: {len(req.imgs)}")

    print(req.imgs[0].height)
    img = bridge.imgmsg_to_cv2(req.imgs[0], "bgr8")

    for i in range(len(req.imgs)):

        # decode images; (ref) http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
#
        try: 
            img = bridge.imgmsg_to_cv2(req.imgs[i], "bgr8")
#
        except CvBridgeError as e:
            sys.exit(e)
#
        cv2.imshow("test", img)
        cv2.waitKey(0)



    """ Response 
    """
    resp = EmergencyDetecResponse()

    resp.task = "Fall Detection"
    resp.state = True   # fall:True,  non-fall: False

    return resp





def detect_emergency_server():
    Node_name = 'fall_detection'
    Bus_name = 'fall_detection'
    rospy.init_node(Node_name)
    s = rospy.Service(Bus_name, EmergencyDetec, fall_detection)


    rospy.spin() # spin() keeps Python from exiting until node is shutdown





if __name__ == "__main__":

    detect_emergency_server()