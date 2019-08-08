#!/usr/bin/env python
###################

# rospy for the publisher
import rospy

# ROS Velocity message
from geometry_msgs.msg import Twist


###################

def move():

    # Define your Velocity topic
 vel_topic = "ford/cmd_vel"
    
    # Set up your Publisher
 rospy.init_node('move', anonymous=True)
 vel_publisher=rospy.Publisher(vel_topic, Twist , queue_size=10)
 vel_msg = Twist()
    
    #Receiveing the user's input
 vel_msg.linear.x = 40.0

     #Since we are moving just in x-axis
 vel_msg.linear.y = 0
 vel_msg.linear.z = 0
 vel_msg.angular.x = 0
 vel_msg.angular.y = 0
 vel_msg.angular.z = 0

 vel_publisher.publish(vel_msg)
    
    # Spin until ctrl + c
 rospy.spin()

###################

if __name__ == '__main__':
    move()
