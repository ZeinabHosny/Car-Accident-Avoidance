#!/usr/bin/env python
###################

# rospy for the publisher
import rospy

# ROS Velocity message
from geometry_msgs.msg import Twist
# ROS Image message
from sensor_msgs.msg import Image

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge
bridge = CvBridge()

count = 0
scenario = input("Input your scenario:")
###################

def image_callback(msg):
 
    vel_topic = "ford/cmd_vel"
    vel_publisher=rospy.Publisher(vel_topic, Twist , queue_size=10)
    vel_msg = Twist()

        #Receiveing the user's input
    vel_msg.linear.x = 30.0

     #Since we are moving just in x-axis
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0

    vel_publisher.publish(vel_msg)
    
    global count
    global scenario
    print("Received an image!")
    print(count)
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
	file_name = str(scenario) + "-"+ str(count) + "_0" + ".jpeg"
        cv2.imwrite(file_name, cv2_img)
        count += 1

###################

def move():

    # Define your Velocity topic
 image_topic = "ford/front_camera/image_raw"

    # Set up your Publisher
 rospy.init_node('move', anonymous=True)
 

  # Set up your Subscriper
 rospy.Subscriber(image_topic, Image, image_callback)
    
    
    # Spin until ctrl + c
 rospy.spin()

###################

if __name__ == '__main__':
    move()
