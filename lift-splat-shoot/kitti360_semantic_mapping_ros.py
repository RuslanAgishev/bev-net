#!/usr/bin/env python

import os
import numpy as np
from pyquaternion import Quaternion
import math
import cv2
from src.tools import denormalize, meters2grid
from tqdm import tqdm
from time import time
import rospy
from nav_msgs.msg import Odometry


class Mapper:
    def __init__(self,
                 dataroot,
                 unknown_prob=0.5,
                 map_radius=400,
                 write_video=False,
                 video_fname='./output.mp4'):
        self.x_lims = [-map_radius, map_radius]
        self.y_lims = [-map_radius, map_radius]
        self.resolution = 0.16
        self.eps = 1e-6
        self.I_sum = 0.
        self.unknown_prob = unknown_prob
        self.rate = rospy.Rate(30)
        self.initialized = False
        self.PATH = dataroot
        self.video_writer = None
        self.video_fname = video_fname
        self.record_video = write_video
        self.T_velo2cam = np.array([[ 0.04307104, -0.99900437, -0.01162549,  0.26234696],
                                    [-0.08829286,  0.00778461, -0.99606414, -0.10763414],
                                    [ 0.99516293,  0.04392797, -0.08786967, -0.82920525],
                                    [ 0.,          0.,          0.,          1.        ],])

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < self.eps

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R) :
        assert(self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < self.eps
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    @staticmethod
    def pred_to_semseg(pred, seed=3):
        # infer the total number of classes along with the spatial dimensions
        # of the mask image via the shape of the output array
        (height, width, numClasses) = pred.shape
        # our output class ID map will be num_classes x height x width in
        # size, so we take the argmax to find the class label with the
        # largest probability for each and every (x, y)-coordinate in the
        # image
        classMap = np.argmax(pred, axis=-1)
        # given the class ID map, we can map each of the class IDs to its
        # corresponding color
        #     np.random.seed(seed)
        #     colors = np.random.randint(0, 255, size=(numClasses-1, 3), dtype=np.uint8)
        #     colors = np.concatenate([colors, 255*np.ones((1, 3), dtype=np.uint8)])
        colors = np.array([
            # [0, 0, 0],
            [128, 64, 128],  # road
            [244, 35, 232],  # sidewalk
            [250, 170, 160],  # parking
            [49, 0, 74],  # dark purple
            [0, 0, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 0, 142]
        ])
        mask = colors[classMap]
        return np.asarray(mask, dtype=np.uint8)

    def pad_map(self, local_map, x_lims, y_lims, resolution):
        x_min, x_max = x_lims
        y_min, y_max = y_lims
        H, W = int((x_max - x_min) / resolution), int((y_max - y_min) / resolution)
        h, w, classes = local_map.shape
        # crop local map in its field of view
        x_axis = np.linspace(0, 2.4, w)
        y_axis = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x_axis, y_axis)
        weights = xx < np.abs(yy)
        local_map[weights, :] = self.unknown_prob
        # local_map = cv2.flip(local_map, 0)
        local_map_expanded = self.unknown_prob * np.ones([H, W, classes])
        local_map_expanded[((H-h)//2):((H+h)//2), W//2:(W+2*w)//2] = local_map
        return local_map_expanded

    @staticmethod
    def publish_odom(pose, orient):
        odom_msg_0 = Odometry()
        odom_msg_0.header.stamp = rospy.Time.now()
        odom_msg_0.header.frame_id = '/odom'
        odom_msg_0.pose.pose.position.x = pose[0]
        odom_msg_0.pose.pose.position.y = pose[1]
        odom_msg_0.pose.pose.position.z = pose[2]
        odom_msg_0.pose.pose.orientation.x = orient.x
        odom_msg_0.pose.pose.orientation.y = orient.y
        odom_msg_0.pose.pose.orientation.z = orient.z
        odom_msg_0.pose.pose.orientation.w = orient.w
        pub = rospy.Publisher('/odom_0', Odometry, queue_size=1)
        pub.publish(odom_msg_0)

    def transform_to_global(self, local_map, pose, yaw_deg):
        x_min, x_max = self.x_lims
        y_min, y_max = self.y_lims
        H, W = int((x_max - x_min) / self.resolution), int((y_max - y_min) / self.resolution)
        # pad local map to fit global map size
        local_map_expanded = self.pad_map(local_map, [x_min, x_max], [y_min, y_max], self.resolution)
        # cv2.imshow('Local map', cv2.resize(local_map_expanded, (600, 600)))

        h, w, classes = local_map.shape
        M = cv2.getRotationMatrix2D((H / 2, W / 2), -yaw_deg, scale=1)
        t = np.array(pose[:2]) // self.resolution
        # t[0] += H/4 - h/2
        t[1] += W / 2 - w / 2
        M[:, 2] += t
        local_map_expanded = cv2.warpAffine(local_map_expanded,
                                            M, (H, W),
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0.5 * np.ones(classes)
                                            )
        return local_map_expanded

    def log_odds_filtering(self, local_map_expanded):
        local_map_expanded = np.clip(local_map_expanded, self.eps, 1 - self.eps)
        I = np.log(local_map_expanded / (1 - local_map_expanded))
        self.I_sum += I
        global_map = 1 / (1 + np.exp(-self.I_sum))
        return global_map

    def add_ego(self, global_map, pose, yaw, dx=69.12, dy=79.36):
        pose_grid = meters2grid(pose[:2],
                                x_lims=self.x_lims, y_lims=self.y_lims,
                                resolution=self.resolution,
                                )
        global_map = cv2.circle(global_map, (pose_grid[0], pose_grid[1]), 8, (0, 255, 0), -1)  # ego pose
        # add fov borders
        bounds = np.array([[0, -dy / 2], [0, dy / 2], [dx, dy / 2], [dx, -dy / 2]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)],
                       [np.sin(yaw), np.cos(yaw)]])
        bounds = Rz @ bounds.T + pose[:2][None].T
        bounds = bounds.T
        bounds_grid = meters2grid(bounds,
                                  x_lims=self.x_lims, y_lims=self.y_lims,
                                  resolution=self.resolution,
                                  )
        global_map = cv2.polylines(global_map, [bounds_grid], True, [0, 1, 0], 3)
        return global_map

    def record(self, grid, filename):
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename,
                                                fourcc, 30,
                                                (grid.shape[1], grid.shape[0]))
        self.video_writer.write(np.asarray(255 * denormalize(grid), dtype=np.uint8))

    def run(self, out_res=800):
        for bev_file, cam2world_file in zip(tqdm(np.sort(os.listdir(os.path.join(self.PATH, 'bev_probs/')))), \
                                            np.sort(os.listdir(os.path.join(self.PATH, 'cam0_to_world/')))):
            if rospy.is_shutdown():
                break
            bev_map = np.load(f'{self.PATH}bev_probs/{bev_file}')
            T_cam2world = np.load(f'{self.PATH}cam0_to_world/{cam2world_file}')
            T = T_cam2world
            # T = self.T_velo2cam @ T_cam2world
            R, t = T[:3, :3], T[:3, 3]
            if not self.initialized:
                t0 = t
                self.initialized = True
            # start from zero origin
            x, y, z = t - t0
            pose = np.array([x, y, z])
            # global orientation
            yaw = self.rotationMatrixToEulerAngles(R)[2] + np.pi / 2
            q = Quaternion(axis=[0, 0, 1], angle=yaw)
            self.publish_odom(pose, q)

            local_map = np.transpose(bev_map[1:4], [1, 2, 0])
            # local_map = np.transpose(bev_map, [1, 2, 0])
            # crop part of the local map
            local_map = local_map[(local_map.shape[0]//2-128):(local_map.shape[0]//2+128), :256, :]
            local_map = cv2.flip(local_map, 0)
            # cv2.imshow('Local map', local_map)

            # apply transformations M=[R, t] in global map frame
            yaw_deg = 180 * yaw / np.pi
            local_map_expanded = self.transform_to_global(local_map,
                                                          pose,
                                                          yaw_deg)
            local_map_expanded = cv2.resize(local_map_expanded, (out_res, out_res))
            # cv2.imshow('Local map', cv2.resize(local_map_expanded, (out_res, out_res)))

            # apply log odds conversion for global map creation and filtering
            global_map = self.log_odds_filtering(local_map_expanded)
            # cv2.imshow('Global map', cv2.resize(global_map, (out_res, out_res)))

            # add ego-pose and local map borders visualization
            # global_map = self.add_ego(global_map, pose, yaw, dx=69.12, dy=79.36)

            global_map_vis = cv2.resize(global_map, (out_res, out_res))
            local_map_vis = cv2.resize(local_map, (out_res, out_res))
            semseg_global = self.pred_to_semseg(global_map_vis)
            semseg_local = self.pred_to_semseg(local_map_vis)
            # result = np.concatenate([semseg_global, semseg_local], axis=1)
            result = global_map_vis

            if self.record_video:
                self.record(result, self.video_fname)

            cv2.imshow('Semantic Mapping', result)
            cv2.waitKey(3)
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('mapping_node')
    proc = Mapper(dataroot='/home/ruslan/datasets/predictions/',
                  unknown_prob=0.5,
                  write_video=False,
                  video_fname='./output_cont.mp4')
    proc.run(out_res=600)
