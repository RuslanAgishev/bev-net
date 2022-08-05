#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField

import numpy as np
import time
from pc_helper import pointcloud2_to_xyz_array
from pc_helper import xyz_array_to_pointcloud2


class PCModifier:
    """
    Subscribes to existing PointCloud2 topic and output a modified PointCloud2
    """

    def __init__(self):
        """
        Initializes node.
        """
        self.pc_sub = rospy.Subscriber("/kitti/velo/pointcloud",
            PointCloud2, self.pc_callback,  queue_size = 1)

    @staticmethod
    def get_pc_sector(pc_array, phi_min, phi_max):
        X, Y, Z = pc_array[:, 0], pc_array[:, 1], pc_array[:, 2]
        # R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(X, Y)
        sector_indexes = np.logical_and(Phi >= phi_min + np.pi / 2., Phi <= phi_max + np.pi / 2.)
        pc_array_sector = pc_array[sector_indexes, :]
        return pc_array_sector

    def pc_callback(self, pc_msg):
        t0 = time.perf_counter()
        pc_array = pointcloud2_to_xyz_array(pc_msg)
        pc_array_sector = self.get_pc_sector(pc_array, phi_min=-np.pi/2., phi_max=np.pi/2.)
        self.publish_pointcloud(pc_array_sector, stamp=pc_msg.header.stamp, topic_name='/velodyne_points')
        print(f'Pointcloud conversion took {1000*(time.perf_counter()-t0):.1f} [ms]')

    def publish_pointcloud(self, points, stamp=None, frame_id='base_scan', topic_name='/velodyne_points_sector'):
        msg = xyz_array_to_pointcloud2(points, stamp=stamp, frame_id=frame_id)
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node('pc_modifier', anonymous=True)
    pc_modifier = PCModifier()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS node")

