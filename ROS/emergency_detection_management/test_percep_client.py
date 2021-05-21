#!/usr/bin/env python

import sys
import os

import rospy

# imports the AddTwoInts service 
from emergency_detection_management.srv import EmergencyDetec, EmergencyDetecRequest
from perception_msgs.srv import persontracking, persontrackingRequest










def test_func(task:str):



    """ Assembling data 
    """

    my_msg = persontrackingRequest()
    my_msg.task = task # add message 





    """ Request & Response 
    """ 

    rospy.wait_for_service('person_detect_tracking_srv')
    
    try:
        # create a handle to the add_two_ints service
        person_tracking = rospy.ServiceProxy('person_detect_tracking_srv', persontracking)     
        print("Requesting msg: ", task )


        # service call 
        res = person_tracking.call(my_msg)

        # service Response 
        print("Response")
        print("state is: ", res.state)
        


    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")








if __name__ == "__main__":

    task = "fall_detection"  # request task message 
    test_func(task)
    
