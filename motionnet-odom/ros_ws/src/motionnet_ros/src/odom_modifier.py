#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry

import numpy as np
import time
from pyquaternion import Quaternion
import math


def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) - np.cos(yaw / 2) * np.sin(pitch / 2) * np.sin(roll / 2)
    qy = np.cos(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) + np.sin(yaw / 2) * np.cos(pitch / 2) * np.sin(roll / 2)
    qz = np.cos(yaw / 2) * np.cos(pitch / 2) * np.sin(roll / 2) - np.sin(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2)
    qw = np.cos(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) + np.sin(yaw / 2) * np.sin(pitch / 2) * np.sin(roll / 2)
    return Quaternion([qw, qx, qy, qz]).normalised


class OdomModifier:
    """
    Subscribes to existing PointCloud2 topic and output a modified PointCloud2
    """

    def __init__(self):
        """
        Initializes node.
        """
        self.vins_odom_sub = rospy.Subscriber("/vins_estimator/odometry", Odometry, self.vins_odom_callback, queue_size=1)
        self.aloam_odom_sub = rospy.Subscriber("/aft_mapped_to_init", Odometry, self.aloam_odom_callback, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom_modified', Odometry, queue_size=1)
        self.loam_yaw_pitch_roll = None

    def vins_odom_callback(self, input_odom):
        x_t = input_odom.pose.pose.position.x
        y_t = input_odom.pose.pose.position.y
        z_t = input_odom.pose.pose.position.z
        translation = np.array([x_t, y_t, z_t])
        x_r = input_odom.pose.pose.orientation.x
        y_r = input_odom.pose.pose.orientation.y
        z_r = input_odom.pose.pose.orientation.z
        w_r = input_odom.pose.pose.orientation.w
        rotation = Quaternion([w_r, x_r, y_r, z_r]).normalised

        output_odom = Odometry()
        output_odom.header = input_odom.header
        q1 = Quaternion(axis=[0, 1, 0], angle=np.pi/2.)
        q2 = Quaternion(axis=[0, 0, 1], angle=-np.pi/2.)
        q3 = q1 * q2  # Composite rotation of q1 then q2 expressed as standard multiplication
        translation = q3.rotate(translation)
        output_odom.pose.pose.position.x = translation[0]
        output_odom.pose.pose.position.y = translation[1]
        output_odom.pose.pose.position.z = translation[2]

        if self.loam_yaw_pitch_roll is not None:
            yaw, pitch, roll = rotation.yaw_pitch_roll
            print('dYaw: ', yaw - self.loam_yaw_pitch_roll[0])
            print('dPitch: ', pitch - self.loam_yaw_pitch_roll[1])
            print('dRoll: ', roll - self.loam_yaw_pitch_roll[2])
            yaw, pitch, roll = self.loam_yaw_pitch_roll
            rotation = euler_to_quaternion(yaw, pitch, roll)

        yaw, pitch, roll = rotation.yaw_pitch_roll
        tmp = euler_to_quaternion(np.pi/4., 0, 0)
        print(tmp)
        yaw, pitch, roll = tmp.yaw_pitch_roll
        print('\nYaw:', yaw, 'Pitch:', pitch, 'Roll:', roll)

        output_odom.pose.pose.orientation.w = tmp[0]
        output_odom.pose.pose.orientation.x = tmp[1]
        output_odom.pose.pose.orientation.y = tmp[2]
        output_odom.pose.pose.orientation.z = tmp[3]

        self.odom_pub.publish(output_odom)

    def aloam_odom_callback(self, input_odom):
        x_t = input_odom.pose.pose.position.x
        y_t = input_odom.pose.pose.position.y
        z_t = input_odom.pose.pose.position.z
        translation = np.array([x_t, y_t, z_t])
        x_r = input_odom.pose.pose.orientation.x
        y_r = input_odom.pose.pose.orientation.y
        z_r = input_odom.pose.pose.orientation.z
        w_r = input_odom.pose.pose.orientation.w
        rotation = Quaternion([w_r, x_r, y_r, z_r]).normalised

        self.loam_yaw_pitch_roll = rotation.yaw_pitch_roll


if __name__ == "__main__":
    rospy.init_node('odom_modifier', anonymous=True)
    odom_modifier = OdomModifier()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node")

