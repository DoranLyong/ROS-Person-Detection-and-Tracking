#!/usr/bin/env python

## Simple demo of a rospy service that add two integers



# import the AddTwoInts service
#from emergency_detection_management.srv import *
from emergency_detection_management.srv import AddTwoInts, AddTwoIntsResponse
import rospy 

def add_two_ints(req):
    print(f"Returning [{req.a} + {req.b} = {req.a + req.b}]")
    return AddTwoIntsResponse(req.a + req.b)

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', AddTwoInts, add_two_ints)

    # spin() keeps Python from exiting until node is shutdown
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()