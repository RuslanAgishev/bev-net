#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
import tf
from cv_bridge import CvBridge, CvBridgeError

import torch
from PIL import Image as pil
import cv2
from pyquaternion import Quaternion
from src.tools import load_config, meters2grid
from src.tools import img_transform, denormalize
from src.tools import normalize_img
from src.models import compile_model
import numpy as np
from time import time


class ImageProcessor:
    def __init__(self,
                 config,
                 model_fpath,
                 image_topic='/cam_front/raw',
                 cam_info_topic='/cam_front/camera_info',
                 odom_topic='/odom',
                 ):
        self.cam_frame = None
        self.img = None
        # nuScenes intrinsics
        self.K = torch.Tensor([[1.2528e+03, 0.0000e+00, 8.2659e+02],
                               [0.0000e+00, 1.2528e+03, 4.6998e+02],
                               [0.0000e+00, 0.0000e+00, 1.0000e+00]])
        self.base_frame = 'base_link'
        self.bridge = CvBridge()
        # self.tl = tf.TransformListener()
        self.cfg = config
        gpuid = self.cfg['TRAIN']['gpuid']
        self.device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
        self.load_model(model_fpath)

        self.image_topic = image_topic
        print("Subscribed to " + self.image_topic)
        img_sub = rospy.Subscriber(image_topic, Image, self.image_callback)

        self.odom_topic = odom_topic
        print("Subscribed to " + self.odom_topic)
        odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)

        if self.K is None:
            print("Subscribed to " + cam_info_topic)
            cam_info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self.cam_info_callback)

        # mapping parameters
        self.I_sum = 0
        self.eps = 1e-6
        self.unknown_prob = 0.5
        self.initialized_odom = False
        self.traj = [np.zeros(2)]
        self.pose = None
        self.orient = None
        self.pose_0 = None
        self.orient_0 = None
        # self.map_lims = {'x': [-50, 50], 'y': [-100, 0]}
        self.map_lims = {'x': [-120, 120], 'y': [-120, 120]}  # currently the global map must be squared
        self.video_writer = None

    def load_model(self, modelf):
        # load the model
        grid_conf = {
            'xbound': self.cfg['DATA']['xbound'],
            'ybound': self.cfg['DATA']['ybound'],
            'zbound': self.cfg['DATA']['zbound'],
            'dbound': self.cfg['DATA']['dbound'],
        }
        data_aug_conf = {
            'resize_lim': self.cfg['DATA']['resize_lim'],
            'final_dim': self.cfg['DATA']['final_dim'],
            'rot_lim': self.cfg['DATA']['rot_lim'],
            'H': self.cfg['DATA']['H'], 'W': self.cfg['DATA']['W'],
            'rand_flip': self.cfg['DATA']['rand_flip'],
            'bot_pct_lim': self.cfg['DATA']['bot_pct_lim'],
            'cams': self.cfg['DATA']['cams'],
            'Ncams': self.cfg['DATA']['ncams'],
        }
        self.model = compile_model(grid_conf, data_aug_conf, outC=2)
        print('loading', modelf)
        self.model.load_state_dict(torch.load(modelf))
        self.model.to(self.device)

    def sample_augmentation(self):
        H, W = self.cfg['DATA']['H'], self.cfg['DATA']['W']
        fH, fW = self.cfg['DATA']['final_dim']

        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.cfg['DATA']['bot_pct_lim']))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def odom_callback(self, odom_msg):
        if not self.initialized_odom:
            self.pose_0 = np.array([odom_msg.pose.pose.position.x,
                                    odom_msg.pose.pose.position.y,
                                    odom_msg.pose.pose.position.z])
            self.orient_0 = Quaternion(x=odom_msg.pose.pose.orientation.x,
                                       y=odom_msg.pose.pose.orientation.y,
                                       z=odom_msg.pose.pose.orientation.z,
                                       w=odom_msg.pose.pose.orientation.w)
            self.initialized_odom =True

        # starting from 0-origin with (pose[0]=[0,0,0], q[0]=[0,0,0,1])
        self.pose = np.array([odom_msg.pose.pose.position.x,
                              odom_msg.pose.pose.position.y,
                              odom_msg.pose.pose.position.z]) - self.pose_0
        # self.pose = self.orient_0.inverse.rotate(self.pose)
        self.orient = Quaternion(x=odom_msg.pose.pose.orientation.x,
                                 y=odom_msg.pose.pose.orientation.y,
                                 z=odom_msg.pose.pose.orientation.z,
                                 w=odom_msg.pose.pose.orientation.w)  # * self.orient_0.inverse
        # q90 = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)
        # self.pose = q90.rotate(self.pose)
        # self.orient = self.orient * q90
        # print(f'Pose: {self.pose[:2]}, Orient: {self.orient.yaw_pitch_roll[0]*180/np.pi}')
        self.traj.append(self.pose[:2])

        # odom_msg_0 = Odometry()
        # odom_msg_0.header = odom_msg.header
        # odom_msg_0.pose.pose.position.x = self.pose[0]
        # odom_msg_0.pose.pose.position.y = self.pose[1]
        # odom_msg_0.pose.pose.position.z = self.pose[2]
        # odom_msg_0.pose.pose.orientation.x = self.orient.x
        # odom_msg_0.pose.pose.orientation.y = self.orient.y
        # odom_msg_0.pose.pose.orientation.z = self.orient.z
        # odom_msg_0.pose.pose.orientation.w = self.orient.w
        # pub = rospy.Publisher('/odom_0', Odometry, queue_size=1)
        # pub.publish(odom_msg_0)
        # print(odom_msg_0.header)

    def image_callback(self, img_msg):
        try:
            if 'compressed' in self.image_topic:
                img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            else:
                img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = pil.fromarray(img)
        self.cam_frame = img_msg.header.frame_id

        if self.K is not None:
            self.run(self.img, self.K)

    def cam_info_callback(self, cam_info_msg):
        fovH = cam_info_msg.height
        fovW = cam_info_msg.width

        K = torch.zeros((3, 3))
        K[0][0] = cam_info_msg.K[0]
        K[0][2] = cam_info_msg.K[2]
        K[1][1] = cam_info_msg.K[4]
        K[1][2] = cam_info_msg.K[5]
        K[2][2] = 1.
        self.K = K.float()

    def preprocessing(self, img, intrin, rot, tran):
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                   resize=resize,
                                                   resize_dims=resize_dims,
                                                   crop=crop,
                                                   flip=flip,
                                                   rotate=rotate,
                                                   )
        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        inputs = [torch.stack([normalize_img(img)]).unsqueeze(0),
                  torch.stack([rot]).unsqueeze(0),
                  torch.stack([tran]).unsqueeze(0),
                  torch.stack([intrin]).unsqueeze(0),
                  torch.stack([post_rot]).unsqueeze(0),
                  torch.stack([post_tran]).unsqueeze(0)]

        return inputs

    def run(self, img, intrin):
        with torch.no_grad():
            # find extrinsics: transformation between base_frame and camera
            # t = self.tl.getLatestCommonTime(self.base_frame, self.cam_frame)
            # tran, quat = self.tl.lookupTransform(self.base_frame, self.cam_frame, t)
            #
            # rot = torch.Tensor(Quaternion(x=quat[0],
            #                               y=quat[1],
            #                               z=quat[2],
            #                               w=quat[3]).normalised.rotation_matrix)
            # tran = torch.Tensor(tran)
            rot = torch.tensor([[5.6848e-03, -5.6367e-03, 9.9997e-01],
                                [-9.9998e-01, -8.3712e-04, 5.6801e-03],
                                [8.0507e-04, -9.9998e-01, -5.6413e-03]])
            tran = torch.tensor([1.7008, 0.0159, 1.5110])

            inputs = self.preprocessing(img, intrin, rot, tran)
            imgs, rots, trans, intrins, post_rots, post_trans = inputs

            # model inference
            t0 = time()
            pred = self.model(imgs.to(self.device),
                             rots.to(self.device),
                             trans.to(self.device),
                             intrins.to(self.device),
                             post_rots.to(self.device),
                             post_trans.to(self.device),
                             )
            dt = time() - t0
            # print(f'Single inference took {dt:.3f} sec \
            #         on input tensor size: {imgs.squeeze().size()}, \
            #         output map size: {pred.squeeze().size()}')
            self.postprocessing(pred)

    def postprocessing(self, pred):
        # postprocessing and visualization
        pred_np = pred.sigmoid().squeeze(0).cpu().numpy()
        local_map = np.transpose(pred_np, (1, 2, 0))
        # add background to predictions
        background = 1 - local_map.sum(axis=-1, keepdims=True)
        local_map = np.concatenate((local_map, background), axis=-1)

        # local map with predictions probabilities
        x_min, x_max = self.map_lims['x']
        y_min, y_max = self.map_lims['y']
        resolution = self.cfg['DATA']['xbound'][2]
        # pad local map to fit global map size
        H, W = int((x_max - x_min) / resolution), int((y_max - y_min) / resolution)
        h, w, classes = local_map.shape
        # crop local map in its field of view
        x_axis = np.linspace(0, 2.4, w)
        y_axis = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x_axis, y_axis)
        weights = xx < np.abs(yy)
        local_map[weights.T, :] = self.unknown_prob
        local_map = cv2.flip(local_map, 0)
        local_map_expanded = cv2.copyMakeBorder(local_map,
                                               (H - 2*h) // 2, H // 2, (W - w) // 2, (W - w) // 2,
                                               cv2.BORDER_CONSTANT,
                                               value=self.unknown_prob*np.ones(classes),
                                               )
        # apply transformations M=[R, t] in global map frame
        yaw = self.orient.yaw_pitch_roll[0]
        yaw_deg = -180*yaw/np.pi - 90
        M = cv2.getRotationMatrix2D((H / 2, W / 2), yaw_deg, 1)
        t = np.array(self.pose[:2]) // resolution
        # t[0] -= W/2
        M[:, 2] += t
        local_map_expanded = cv2.warpAffine(local_map_expanded,
                              M, (H, W),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=self.unknown_prob*np.ones(classes))
        # print(np.unique(grid))
        # np.savez(f'./grids/grid_{time()}.npz', grid=grid)

        # # add trajectory
        # traj = np.array(self.traj)
        # traj_grid = meters2grid(traj)-1
        # grid[traj_grid[:, 1], traj_grid[:, 0], :] = 1.

        # apply log odds conversion for global map creation and filterring
        local_map_expanded = np.clip(local_map_expanded, self.eps, 1-self.eps)
        I = np.log(local_map_expanded / (1 - local_map_expanded))
        self.I_sum += I
        global_map = 1 / (1 + np.exp(-self.I_sum))
        # add ego-pose and local map borders visualization
        pose_grid = meters2grid(self.pose[:2],
                               x_lims=self.map_lims['x'], y_lims=self.map_lims['y'],
                               resolution=self.cfg['DATA']['xbound'][2]
                               )
        cv2.circle(global_map, (pose_grid[0], pose_grid[1]), 8, (0, 255, 0), -1)  # ego pose
        dy = self.cfg['DATA']['xbound'][1] - self.cfg['DATA']['xbound'][0]
        dx = self.cfg['DATA']['ybound'][1] - self.cfg['DATA']['ybound'][0]
        bounds = np.array([[0, -dy / 2], [0, dy / 2], [dx, dy / 2], [dx, -dy / 2]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)],
                       [np.sin(yaw),  np.cos(yaw)]])
        bounds = Rz @ bounds.T + self.pose[:2][None].T
        bounds = bounds.T
        bounds_grid = meters2grid(bounds,
                                  x_lims=self.map_lims['x'], y_lims=self.map_lims['y'],
                                  resolution=self.cfg['DATA']['xbound'][2]
                                  )
        cv2.polylines(global_map, [bounds_grid], True, [0, 1, 0], 3)

        img_vis = denormalize(cv2.resize(np.asarray(self.img)[..., (2, 1, 0)], (800, 450)))
        global_map_vis = cv2.flip( cv2.resize(global_map, (450, 450)), 1 )
        local_map_vis = cv2.flip(cv2.resize(local_map, (450, 450)), 1)
        # result = np.concatenate([img_vis, global_map_vis, local_map_vis], axis=1)
        result = np.concatenate([img_vis, global_map_vis], axis=1)
        cv2.imshow('LSS Semantic Mapping', result)
        # if self.video_writer is None:
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     self.video_writer = cv2.VideoWriter('./output.mp4',
        #                                         fourcc, 10,
        #                                         (result.shape[1], result.shape[0]))
        # self.video_writer.write(np.asarray(255*denormalize(result), dtype=np.uint8))
        # cv2.imshow('Image', img_vis)
        # cv2.imshow('Road', cv2.flip(global_map[..., 0], 1))
        # cv2.imshow('Cars', cv2.flip(global_map[..., 1], 1))
        # cv2.imshow('Global map', cv2.flip( cv2.resize(global_map, (450, 450)), 1 ))
        # cv2.imshow('I', cv2.flip(I, 1))
        # cv2.imshow('Local map', cv2.flip(local_map, 1))
        cv2.waitKey(3)


if __name__ == '__main__':
    rospy.init_node('lss_node')
    cfg = load_config('./configs/nuscenes_config.yaml')
    proc_nusc = ImageProcessor(cfg,
                              model_fpath='./weights/road_cars_2discr/model_iou_0.8142298854050972.pt',
                              # image_topic=rospy.get_param('~image_topic', '/cam_front/raw'),
                              image_topic=rospy.get_param('~image_topic', '/camera/image_raw'),
                              # odom_topic=rospy.get_param('~odom_topic', '/orb_slam3/odom'),
                              odom_topic=rospy.get_param('~odom_topic', '/odom'),
                              )
    rospy.spin()
