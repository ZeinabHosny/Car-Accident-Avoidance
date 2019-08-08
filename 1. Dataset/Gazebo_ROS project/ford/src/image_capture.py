#! /usr/bin/python
###################

# rospy for the subscriber
import rospy

# ROS Image message
from sensor_msgs.msg import Image

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge
bridge = CvBridge()

###################

count = 0

def image_callback(msg):
    global count
    print("Received an image!")
    print(count)
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
	file_name = "camera_image" + str(count) + "_0" + ".jpeg"
        cv2.imwrite(file_name, cv2_img)
        count += 1

###################

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "ford/front_camera/image_raw"
    
    
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
