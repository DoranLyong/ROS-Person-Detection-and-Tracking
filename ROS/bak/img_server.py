#!/opt/conda/bin/python

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


    np_bbox = np.asarray(req.bboxes).reshape(-1, 4) # decode bbox order [x1, y2, x2, y2]
    print(np_bbox)

    print("Received bbox_seq length: ", np_bbox.shape[0])  # (N, 4) bbox object 
    print("Received img_seq length:" , len(req.imgs))



#    img_decode = bridge.imgmsg_to_cv2(req.imgs[0], "bgr8")
#    cv2.imwrite("./test_test/decode.jpg", img_decode)


#    for i in range(len(req.imgs)):
#
#        # decode images; (ref) http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
##
#        try: 
#            img = bridge.imgmsg_to_cv2(req.imgs[i], "bgr8")
##
#        except CvBridgeError as e:
#            sys.exit(e)
##
#        cv2.imshow("test", img)
#        cv2.waitKey(0)
#
#        print("bbox is : ", np_bbox[i, :])
#


    """ Response 
    """
    resp = EmergencyDetecResponse()

    resp.task = "Fall Detection"
    resp.state = False   # fall:True,  non-fall: False

    return resp





def detect_emergency_server():
    Node_name = 'fall_detection'
    Bus_name = 'fall_detection'
    rospy.init_node(Node_name)
    s = rospy.Service(Bus_name, EmergencyDetec, fall_detection)


    rospy.spin() # spin() keeps Python from exiting until node is shutdown





if __name__ == "__main__":

    detect_emergency_server()