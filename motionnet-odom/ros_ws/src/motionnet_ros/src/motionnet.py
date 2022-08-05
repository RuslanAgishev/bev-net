#!/usr/bin/env python

# MotionNet import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
PATH_TO_MOTIONNET='/home/ruslan/Desktop/DoEdu/src/bev-net/motionnet-odom/'
# PATH_TO_MOTIONNET='/root/motionnet/'
sys.path.append(PATH_TO_MOTIONNET)
import torch
import torch.nn as nn
import numpy as np
import time
from model import MotionNet
import matplotlib.pyplot as plt
from data.data_utils import voxelize_occupy
from data.data_utils import voxelize_occupy_gpu, gen_non_empty_map_gpu
from utils import visualize_prediction
from pyquaternion import Quaternion
import cupy as cp # For CUDA 10.2: pip install cupy-cuda102
from collections import deque
from copy import deepcopy
from functools import reduce

# ROS import
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from pc_helper import pointcloud2_to_array, get_pc_sector, publish_pointcloud


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0, 0, 1], radians=yaw)


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


class Processor_ROS:
    def __init__(self, model_path, pointcloud_topic='/velodyne_points', odom_topic='/odom'):
        self.model_path = model_path
        self.device = None
        self.model = None
        self.pointcloud_topic = pointcloud_topic
        self.odom_topic = odom_topic
        self.voxel_size = (0.25, 0.25, 0.4)
        self.area_extents = np.array([[-32., 32.], [-32., 32.], [-3, 2]])
        self.predictions = {}
        self.lidar_deque = deque(maxlen=5)
        self.current_frame = {
            "lidar_stamp": None,
            "lidar_seq": None,
            "points": None,
            "odom_seq": None,
            "odom_stamp": None,
            "translation": None,
            "rotation": None
        }
        self.pc_list = deque(maxlen=5)
        self.inputs = None
        self.predictions = {}
        self.publish_sync_pointcloud = rospy.get_param('~/motionnet/publish_synch_pcs')
        self.visualize_predictions = PLT_VISUALIZATION
        self.frame_ind = 0

    def initialize(self):
        # # nuscenes dataset
        # lidar2imu_t = np.array([0.985793, 0.0, 1.84019])
        # lidar2imu_r = Quaternion([0.706749235, -0.01530099378, 0.0173974518, -0.7070846])
        lidar2imu_t = np.array([0.0, 0.0, 0.0])
        lidar2imu_r = Quaternion([1.0, 0.0, 0.0, 0.0])
        self.lidar2imu = transform_matrix(lidar2imu_t, lidar2imu_r, inverse=True)
        self.imu2lidar = transform_matrix(lidar2imu_t, lidar2imu_r, inverse=False)

        self.load_model()
        self.sub_lidar = rospy.Subscriber(
            self.pointcloud_topic, PointCloud2, self.rslidar_callback, queue_size=1, buff_size=2 ** 24)
        print('[INFO] Subscribing to', self.pointcloud_topic)
        self.sub_odom = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=10, buff_size=2 ** 10, tcp_nodelay=True)
        print('[INFO] Subscribing to', self.odom_topic)
        self.pub_sync_cloud = rospy.Publisher("sync_5sweeps_cloud", PointCloud2, queue_size=1)

    def load_model(self):
        # MotionNet model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=13)
        self.model = nn.DataParallel(self.model)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def run(self):
        # Inference
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            inference_outputs = self.model(self.inputs)
        torch.cuda.synchronize()
        print(f"Inference time cost: {1000*(time.time() - t0):.1f} [ms]")
        # Postprocessing
        t1 = time.time()
        disp_pred, cat_pred, motion_pred = self.motionnet_postprocessing_gpu(inference_outputs)
        print(f'Postprocessing took {1000*(time.time()-t1):.1f} [ms]')
        return disp_pred, cat_pred, motion_pred

    def motionnet_postprocessing_gpu(self, inference_outputs):
        disp_pred, cat_pred, motion_pred = inference_outputs

        disp_pred = cp.asarray(disp_pred.cpu().numpy())
        disp_pred = cp.transpose(disp_pred, (0, 2, 3, 1))
        cat_pred = cp.asarray(cat_pred.cpu().numpy())
        cat_pred = cp.squeeze(cat_pred, 0)
        motion_pred = cp.asarray( motion_pred.cpu().numpy() )
        # print(f'Copy from GPU to CPU memory took {1000*(time.time()-t1):.1f} [ms]')
        # The prediction are the displacement between adjacent frames
        for c in range(1, disp_pred.shape[0]):
            disp_pred[c, ...] = disp_pred[c, ...] + disp_pred[c - 1, ...]

        motion_pred = cp.argmax(motion_pred, axis=1)
        motion_mask = motion_pred == 0

        last_pc = self.pc_list[-1].T
        non_empty_map = gen_non_empty_map_gpu(last_pc, cp.asarray(self.voxel_size[0:2]), cp.asarray(self.area_extents))
        cat_mask = cp.logical_and( cp.argmax(cat_pred, axis=0) == 0, non_empty_map == 1 )
        cat_mask = cp.expand_dims(cat_mask, 0)
        cat_weight_map = cp.ones_like(motion_pred, dtype=cp.float32)
        cat_weight_map[motion_mask] = 0.0
        cat_weight_map[cat_mask] = 0.0
        cat_weight_map = cat_weight_map[:, :, :, cp.newaxis]  # (1, h, w, 1)

        disp_pred = disp_pred * cat_weight_map
        cat_pred = cp.argmax(cat_pred, axis=0) + 1
        cat_pred = (cat_pred * non_empty_map).astype(cp.int)
        return disp_pred, cat_pred, motion_pred

    def get_lidar_and_odom_data(self, input_points: dict):
        print("Got one frame lidar data.")
        self.current_frame["lidar_stamp"] = input_points['stamp']
        self.current_frame["lidar_seq"] = input_points['seq']
        self.current_frame["points"] = input_points['points'].T
        self.lidar_deque.append(deepcopy(self.current_frame))

        # Check if enough N=5 of input lidar clouds is collected and odometry is received
        if len(self.lidar_deque) == 5 and self.lidar_deque[0]['translation'] is not None:
            t0 = time.time()

            ref_from_car = self.imu2lidar
            car_from_global = transform_matrix(self.lidar_deque[-1]['translation'], self.lidar_deque[-1]['rotation'],
                                               inverse=True)

            ref_from_car_gpu = cp.asarray(ref_from_car)
            car_from_global_gpu = cp.asarray(car_from_global)

            voxels_list = list()
            for i in range(len(self.lidar_deque) - 1):
                last_pc = self.lidar_deque[i]['points']
                last_pc_gpu = cp.asarray(last_pc)

                global_from_car = transform_matrix(self.lidar_deque[i]['translation'], self.lidar_deque[i]['rotation'],
                                                   inverse=False)
                car_from_current = self.lidar2imu
                global_from_car_gpu = cp.asarray(global_from_car)
                car_from_current_gpu = cp.asarray(car_from_current)

                transform = reduce(
                    cp.dot,
                    [ref_from_car_gpu, car_from_global_gpu, global_from_car_gpu, car_from_current_gpu],
                )
                # tmp_1 = cp.dot(global_from_car_gpu, car_from_current_gpu)
                # tmp_2 = cp.dot(car_from_global_gpu, tmp_1)
                # transform = cp.dot(ref_from_car_gpu, tmp_2)

                last_pc_gpu = cp.vstack((last_pc_gpu[:3, :], cp.ones(last_pc_gpu.shape[1])))
                last_pc_gpu = cp.dot(transform, last_pc_gpu)

                self.pc_list.append(last_pc_gpu[:3, :])

                # Create voxels out of pointcloud
                voxels = voxelize_occupy_gpu(last_pc_gpu[:4, :].T, voxel_size=cp.asarray(self.voxel_size),
                                             extents=cp.asarray(self.area_extents))
                voxels_list.append(cp.asarray(voxels))

            current_pc = self.lidar_deque[-1]['points']
            current_pc_gpu = cp.asarray(current_pc)
            self.pc_list.append(current_pc_gpu[:3, :])

            # Create voxels out of pointcloud
            voxels = voxelize_occupy_gpu(current_pc_gpu[:4, :].T, voxel_size=cp.asarray(self.voxel_size),
                                         extents=cp.asarray(self.area_extents))
            voxels_list.append(cp.asarray(voxels))

            padded_voxel_points = torch.from_numpy(cp.asnumpy(cp.stack(voxels_list, axis=0)))
            self.inputs = torch.unsqueeze(padded_voxel_points, 0)
            print(f'PC sequence preprocessing took {1000 * (time.time() - t0):.1f} [ms]')

            if self.publish_sync_pointcloud:
                # Reference: https://github.com/tianweiy/CenterPoint/blob/master/tools/multi_sweep_inference.py#L250
                all_pc = np.zeros((5, 0), dtype=float)
                for i in range(len(self.pc_list)):
                    tmp_pc = cp.vstack((self.pc_list[i], cp.zeros((2, self.pc_list[i].shape[1]))))
                    tmp_pc = cp.asnumpy(tmp_pc)
                    ref_timestamp = self.lidar_deque[-1]['lidar_stamp'].to_sec()
                    timestamp = self.lidar_deque[i]['lidar_stamp'].to_sec()
                    tmp_pc[3, ...] = self.lidar_deque[i]['points'][3, ...]
                    tmp_pc[4, ...] = ref_timestamp - timestamp
                    all_pc = np.hstack((all_pc, tmp_pc))

                all_pc = all_pc.T
                print(f"Concate pointcloud shape: {all_pc.shape}")
                sync_cloud = self.xyz_array_to_pointcloud2(all_pc[:, :3], stamp=self.lidar_deque[-1]["lidar_stamp"],
                                                           frame_id="velo_link")
                self.pub_sync_cloud.publish(sync_cloud)
            return True

    def get_odom_data(self, input_odom):
        self.current_frame["odom_stamp"] = input_odom.header.stamp
        self.current_frame["odom_seq"] = input_odom.header.seq
        x_t = input_odom.pose.pose.position.x
        y_t = input_odom.pose.pose.position.y
        z_t = input_odom.pose.pose.position.z
        # if 'vins' in self.odom_topic:
        #     print('visual odometry transforms')
        #     self.current_frame["translation"] = np.array([z_t, y_t, x_t])
        # else:
        #     self.current_frame["translation"] = np.array([x_t, y_t, z_t])
        self.current_frame["translation"] = np.array([x_t, y_t, z_t])
        x_r = input_odom.pose.pose.orientation.x
        y_r = input_odom.pose.pose.orientation.y
        z_r = input_odom.pose.pose.orientation.z
        w_r = input_odom.pose.pose.orientation.w
        self.current_frame["rotation"] = Quaternion([w_r, x_r, y_r, z_r])

    @staticmethod
    def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
        '''
        '''
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(
                cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]

        points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
        points[..., 0] = cloud_array['x']
        points[..., 1] = cloud_array['y']
        points[..., 2] = cloud_array['z']
        try:
            points[..., 3] = cloud_array['i']
        except:
            pass
        return points
    @staticmethod
    def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array of points.
        '''
        msg = PointCloud2()
        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = points_sum.shape[0]
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
            # PointField('i', 12, PointField.FLOAT32, 1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = points_sum.shape[0]
        msg.is_dense = int(np.isfinite(points_sum).all())
        msg.data = np.asarray(points_sum, np.float32).tostring()
        # msg.data = points_sum.astype(np.float32).tobytes()
        return msg

    def rslidar_callback(self, msg):
        t_t = time.time()
        msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        np_p = self.get_xyz_points(msg_cloud, remove_nans=True)
        # np_p = get_pc_sector(np_p, -np.pi/4., np.pi/4.)
        # publish_pointcloud(np_p[:, :3], stamp=msg.header.stamp)

        print("  ")
        seq = msg.header.seq
        stamp = msg.header.stamp
        input_points = {
            'stamp': stamp,
            'seq': seq,
            'points': np_p
        }
        if (self.get_lidar_and_odom_data(input_points)):
            disp_pred, cat_pred, motion_pred = self.run()
            self.predictions['displacement'] = disp_pred
            self.predictions['category'] = cat_pred
            self.predictions['motion'] = motion_pred

            if self.visualize_predictions:
                plt.cla()
                t_vis_start = time.time()
                result_path = None #os.path.join(PATH_TO_MOTIONNET, f'test/{self.frame_ind}.png')
                visualize_prediction(cp.asnumpy(self.predictions['category']),
                                     cp.asnumpy(self.predictions['displacement']),
                                     self.voxel_size,
                                     file_savepath=result_path)
                dt = time.time() - t_vis_start
                print(f'Visualization took {1000 * dt:.1f} [ms]')
                self.frame_ind += 1
                plt.pause(dt)

            print(f'\n--- PC callback run time: {1000 * (time.time() - t_t):.1f} [ms] ---\n')

    def odom_callback(self, msg):
        '''
        get odom data
        '''
        self.get_odom_data(msg)


PLT_VISUALIZATION = rospy.get_param('~/motionnet/plt_visualization')

if __name__ == "__main__":
    rospy.init_node('motionnet_ros_node')

    proc = Processor_ROS(model_path=PATH_TO_MOTIONNET+'weights/model.pth',
                         pointcloud_topic=rospy.get_param('~/motionnet/pointcloud_topic'),
                         odom_topic=rospy.get_param('~/motionnet/odom_topic'))
    proc.initialize()
    print("[INFO] MotionNet ros_node has started!")

    rospy.spin()
