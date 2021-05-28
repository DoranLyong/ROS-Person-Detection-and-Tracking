#!/usr/bin/env python3

#%%
import sys
import os

import cv2
import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError






#%%  Getting RealSense image frame in ROS
""" (ref) https://stackoverflow.com/questions/62938146/getting-realsense-depth-frame-in-ros
"""


class ImageListener(object):
    def __init__(self, topic):
        self.cv2_img = None
        
        self.topic = topic
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, Image , self.imageCallback)

        

    def imageCallback(self, img_msg):

        try:
            self.cv2_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')   # Cvt ROS image msg to cv2 img 
                                                                                        # (ref) http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

            cv2.imshow("test", self.cv2_img)
            cv2.waitKey(32)

        except CvBridgeError as e:
            print(e)
            return



#%% 
if __name__ == '__main__':

    """ For usage test 
    """ 
    rospy.init_node("ROS_image_processor")
    topic = '/camera/color/image_raw'  # check the depth image topic in your Gazebo environmemt and replace this with your
    listener = ImageListener(topic)

    rospy.spin()