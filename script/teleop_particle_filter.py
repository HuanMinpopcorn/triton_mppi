#!/usr/bin/python3

# This script is used to teleoperate the particle filter using keyboard inputs.
# It allows the user to control the robot's movement and adjust the particle filter parameters in real-time.
# The script uses the keyboard to send commands to the robot and the particle filter.

# The user can control the robot's movement using the 'w', 'a', 's', and 'd' keys.
# w: move forward
# a: turn left
# s: move backward
# d: turn right

import rospy
from geometry_msgs.msg import Twist
import numpy as np

SPD = 1.5  # Speed of the robot
ANG = np.pi/2  # Angular speed of the robot

def teleop_particle_filter():
    rospy.init_node('teleop_particle_filter', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    twist = Twist()

    # Setup the initial twist message
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0

    print("Teleop Particle Filter")
    print("Use 'w', 'a', 's', 'd' to control the robot")
    print("Press 'q' to quit")

    while not rospy.is_shutdown():
        key = input("Enter command: ")
        if key == 'w':
            twist.linear.x = SPD
            twist.angular.z = 0.0
        elif key == 'a':
            twist.linear.x = 0.0
            twist.angular.z = ANG
        elif key == 's':
            twist.linear.x = -SPD
            twist.angular.z = 0.0
        elif key == 'd':
            twist.linear.x = 0.0
            twist.angular.z = -ANG
        elif key == 'q':
            break
        else:
            print("Invalid command")

        pub.publish(twist)
        rate.sleep()
    # Reset the twist message to stop the robot
    twist.linear.x = 0.0
    twist.angular.z = 0.0
    pub.publish(twist)
    print("Stopping the robot")
    print("Exiting teleop_particle_filter")

if __name__ == '__main__':
    try:
        teleop_particle_filter()
    except rospy.ROSInterruptException:
        pass