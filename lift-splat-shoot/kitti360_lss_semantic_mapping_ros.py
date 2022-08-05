#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import tf
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from nav_msgs.msg import Odometry

import torch
from PIL import Image as pil
import cv2
from pyquaternion import Quaternion
from src.tools import load_config
from src.tools import img_transform, denormalize
from src.tools import normalize_img
from src.models import compile_model
import numpy as np
import time


class ImageProcessor:
    def __init__(self,
                 config,
                 model_fpath,
                 image_left_topic,
                 image_right_topic,
                 cam_left_info_topic,
                 cam_right_info_topic,
                 odom_topic='/odom',
                 map_radius=400,
                 ):
        self.base_frame = 'base_link'
        self.bridge = CvBridge()
        self.cfg = config
        gpuid = self.cfg['TRAIN']['gpuid']
        self.device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
        self.load_model(model_fpath)
        self.tl = tf.TransformListener()

        self.img_left_msg = None
        self.img_right_msg = None
        self.cam_left_msg = None
        self.cam_right_msg = None
        self.odom_msg = None
        print(f'Subscribing to {image_left_topic}')
        rospy.Subscriber(image_left_topic, Image, self.img_left_callback)
        print(f'Subscribing to {image_right_topic}')
        rospy.Subscriber(image_right_topic, Image, self.img_right_callback)
        print(f'Subscribing to {cam_left_info_topic}')
        rospy.Subscriber(cam_left_info_topic, CameraInfo, self.cam_left_callback)
        print(f'Subscribing to {cam_right_info_topic}')
        rospy.Subscriber(cam_right_info_topic, CameraInfo, self.cam_right_callback)
        print(f'Subscribing to {odom_topic}')
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback)

        # mapping parameters
        self.x_lims = [-map_radius, map_radius]
        self.y_lims = [-map_radius, map_radius]
        self.resolution = 0.16
        self.unknown_prob = 0.5
        self.eps = 1e-6
        self.I_sum = 0.
        # odometry parameters
        self.pose = None
        self.orient = None
        self.pose_0 = None
        self.orient_0 = None
        self.initialized_odom = False

        self.video_writer = None

        # print(f'Subscribing to {image_left_topic}')
        # image_left_sub = message_filters.Subscriber(image_left_topic, Image)
        # print(f'Subscribing to {image_right_topic}')
        # image_right_sub = message_filters.Subscriber(image_right_topic, Image)
        # print(f'Subscribing to {cam_left_info_topic}')
        # cam_left_info_sub = message_filters.Subscriber(cam_left_info_topic, CameraInfo)
        # print(f'Subscribing to {cam_right_info_topic}')
        # cam_right_info_sub = message_filters.Subscriber(cam_right_info_topic, CameraInfo)
        #
        # ts = message_filters.ApproximateTimeSynchronizer([image_left_sub,
        #                                                   image_right_sub,
        #                                                   cam_left_info_sub,
        #                                                   cam_right_info_sub],
        #                                                   queue_size=1, slop=0.1, allow_headerless=True)
        # ts.registerCallback(self.run)

    def img_left_callback(self, msg):
        self.img_left_msg = msg

    def img_right_callback(self, msg):
        self.img_right_msg = msg

    def cam_left_callback(self, msg):
        self.cam_left_msg = msg

    def cam_right_callback(self, msg):
        self.cam_right_msg = msg

    def odom_callback(self, odom_msg):
        self.odom_msg = odom_msg
        if not self.initialized_odom:
            self.pose_0 = np.array([odom_msg.pose.pose.position.x,
                                    odom_msg.pose.pose.position.y,
                                    odom_msg.pose.pose.position.z])
            self.orient_0 = Quaternion(x=odom_msg.pose.pose.orientation.x,
                                       y=odom_msg.pose.pose.orientation.y,
                                       z=odom_msg.pose.pose.orientation.z,
                                       w=odom_msg.pose.pose.orientation.w).normalised
            self.initialized_odom = True

        # starting from 0-origin with (pose[0]=[0,0,0], q[0]=[0,0,0,1])
        self.pose = np.array([odom_msg.pose.pose.position.x,
                              odom_msg.pose.pose.position.y,
                              odom_msg.pose.pose.position.z]) - self.pose_0
        self.pose = self.orient_0.inverse.rotate(self.pose)
        self.orient = Quaternion(x=odom_msg.pose.pose.orientation.x,
                                 y=odom_msg.pose.pose.orientation.y,
                                 z=odom_msg.pose.pose.orientation.z,
                                 w=odom_msg.pose.pose.orientation.w).normalised * self.orient_0.inverse

    def ground_truth_odometry(self, origin_frame="/world", base_frame="/base_link"):
        self.tl.waitForTransform(origin_frame, base_frame, rospy.Time(0), rospy.Duration(1))
        position, quat = self.tl.lookupTransform(origin_frame, base_frame, rospy.Time(0))
        if not self.initialized_odom:
            self.pose_0 = np.array(position)
            self.orient_0 = Quaternion(x=quat[0],
                                 y=quat[1],
                                 z=quat[2],
                                 w=quat[3]).normalised
            self.initialized_odom = True
        self.pose = np.array(position) - self.pose_0
        self.pose = self.orient_0.inverse.rotate(self.pose)
        self.orient = Quaternion(x=quat[0],
                                 y=quat[1],
                                 z=quat[2],
                                 w=quat[3]).normalised * self.orient_0.inverse
        self.publish_odom(self.pose, self.orient)
        return self.pose, self.orient

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

    def img_msg_to_pil(self, img_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return pil.fromarray(img)

    def cam_info_msg_to_tensor(self, cam_info_msg):
        K = torch.zeros((3, 3))
        K[0][0] = cam_info_msg.K[0]
        K[0][2] = cam_info_msg.K[2]
        K[1][1] = cam_info_msg.K[4]
        K[1][2] = cam_info_msg.K[5]
        K[2][2] = 1.
        return K.float()

    def get_extrinsics(self, cam_frame):
        # find transformation between base_frame and camera
        t = self.tl.getLatestCommonTime(self.base_frame, cam_frame)
        tran, quat = self.tl.lookupTransform(self.base_frame, cam_frame, t)

        rot = torch.Tensor(Quaternion(x=quat[0],
                                      y=quat[1],
                                      z=quat[2],
                                      w=quat[3]).normalised.rotation_matrix)
        tran = torch.Tensor(tran)
        return tran, rot

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

    def preprocessing(self, img):
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
        return normalize_img(img), post_rot, post_tran

    def postprocessing(self, pred):
        # postprocessing and visualization
        pred_np = pred.sigmoid().squeeze(0).cpu().numpy()
        local_map = np.transpose(pred_np, (1, 2, 0))
        # add background to predictions
        background = 1 - local_map.sum(axis=-1, keepdims=True)
        local_map = np.concatenate((local_map, background), axis=-1)
        return local_map

    @staticmethod
    def publish_odom(pose, orient, frame_id='/world'):
        odom_msg_0 = Odometry()
        odom_msg_0.header.stamp = rospy.Time.now()
        odom_msg_0.header.frame_id = frame_id
        odom_msg_0.pose.pose.position.x = pose[0]
        odom_msg_0.pose.pose.position.y = pose[1]
        odom_msg_0.pose.pose.position.z = pose[2]
        odom_msg_0.pose.pose.orientation.x = orient.x
        odom_msg_0.pose.pose.orientation.y = orient.y
        odom_msg_0.pose.pose.orientation.z = orient.z
        odom_msg_0.pose.pose.orientation.w = orient.w
        pub = rospy.Publisher('/odom_0', Odometry, queue_size=1)
        pub.publish(odom_msg_0)

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
        # add padding with self.unknown_prob values
        local_map_expanded = self.unknown_prob * np.ones([H, W, classes])
        local_map_expanded[(H//2):(H//2+h), (W//2-w//2):(W//2+w//2)] = local_map
        return local_map_expanded

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
        t[1] += W//3
        M[:, 2] += t
        local_map_expanded = cv2.warpAffine(local_map_expanded,
                                            M, (H, W),
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=self.unknown_prob * np.ones(classes)
                                            )
        return local_map_expanded

    def log_odds_filtering(self, local_map_expanded):
        local_map_expanded = np.clip(local_map_expanded, self.eps, 1 - self.eps)
        I = np.log(local_map_expanded / (1 - local_map_expanded))
        self.I_sum += I
        global_map = 1 / (1 + np.exp(-self.I_sum))
        return global_map

    def processor(self,
                  img_left_msg,
                  img_right_msg,
                  cam_left_info_msg,
                  cam_right_info_msg,
                  out_res=500):
        with torch.no_grad():
            img_left = self.img_msg_to_pil(img_left_msg)
            img_right = self.img_msg_to_pil(img_right_msg)

            img_norm_left, post_rot_left, post_tran_left = self.preprocessing(img_left)
            img_norm_right, post_rot_right, post_tran_right = self.preprocessing(img_right)

            K_left = self.cam_info_msg_to_tensor(cam_left_info_msg)
            K_right = self.cam_info_msg_to_tensor(cam_right_info_msg)

            tran_left, rot_left = self.get_extrinsics(img_left_msg.header.frame_id)
            tran_right, rot_right = self.get_extrinsics(img_right_msg.header.frame_id)

            inputs = [torch.stack([img_norm_left, img_norm_right]).unsqueeze(0),
                      torch.stack([rot_left, rot_right]).unsqueeze(0),
                      torch.stack([tran_left, tran_right]).unsqueeze(0),
                      torch.stack([K_left, K_right]).unsqueeze(0),
                      torch.stack([post_rot_left, post_rot_right]).unsqueeze(0),
                      torch.stack([post_tran_left, post_tran_right]).unsqueeze(0)]
            imgs, rots, trans, intrins, post_rots, post_trans = inputs

            # model inference
            t0 = time.time()
            pred = self.model(imgs.to(self.device),
                              rots.to(self.device),
                              trans.to(self.device),
                              intrins.to(self.device),
                              post_rots.to(self.device),
                              post_trans.to(self.device),
                              )
            dt = time.time() - t0
            # print(f'Single inference took {dt:.3f} sec \
            #                     on input tensor size: {imgs.squeeze().size()}, \
            #                     output map size: {pred.squeeze().size()}')
            # print()
            local_map = self.postprocessing(pred)
            local_map = cv2.flip(local_map, 0)
            local_map = cv2.rotate(local_map, cv2.cv2.ROTATE_90_CLOCKWISE)

            # Mapping with odometry input
            if self.odom_msg is None:
                pose, quat = self.ground_truth_odometry()
            else:
                pose, quat = self.pose, self.orient
            yaw = quat.yaw_pitch_roll[0]

            # apply transformations M=[R, t] in global map frame
            yaw_deg = 180 * yaw / np.pi
            local_map_expanded = self.transform_to_global(local_map,
                                                          pose,
                                                          yaw_deg)
            local_map_expanded = cv2.resize(local_map_expanded, (out_res, out_res))
            # cv2.imshow('Local map', local_map_expanded)

            # apply log odds conversion for global map creation and filtering
            global_map = self.log_odds_filtering(local_map_expanded)

            # Visualization
            global_map_vis = cv2.resize(global_map, (out_res, out_res))
            img_left_vis = denormalize(cv2.resize(np.asarray(img_left)[..., (2, 1, 0)], (out_res, out_res//2)))
            img_right_vis = denormalize(cv2.resize(np.asarray(img_right)[..., (2, 1, 0)], (out_res, out_res//2)))
            imgs_vis = np.concatenate([img_left_vis, img_right_vis], axis=1)
            local_map_vis = cv2.resize(local_map, (out_res, out_res))
            maps_vis = np.concatenate([global_map_vis, local_map_vis], axis=1)
            result = np.concatenate([imgs_vis, maps_vis], axis=0)
            cv2.imshow('Result', result)
            # cv2.imshow('Images', imgs_vis)
            # cv2.imshow('Local map', local_map_vis)
            # cv2.imshow('Global map', global_map_vis)
            # cv2.imshow('Cars', local_map_vis[..., 1])
            cv2.waitKey(3)
            # write video
            # self.record(result, 'output.mp4')

    def record(self, frame, filename):
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename,
                                                fourcc, 10,
                                                (frame.shape[1], frame.shape[0]))
        self.video_writer.write(np.asarray(255 * denormalize(frame), dtype=np.uint8))

    def run(self, max_delay=0.05):
        if self.img_left_msg is not None and self.img_right_msg is not None and \
                self.cam_left_msg is not None and self.cam_right_msg is not None:
            imgs_delay = (self.img_left_msg.header.stamp - self.img_right_msg.header.stamp).to_sec()
            if imgs_delay < max_delay:
                self.processor(self.img_left_msg,
                               self.img_right_msg,
                               self.cam_left_msg,
                               self.cam_right_msg)


if __name__ == '__main__':
    rospy.init_node('lss_node')
    cfg = load_config('./configs/kitti360_config.yaml')
    proc = ImageProcessor(cfg,
                          model_fpath='./weights/road_cars_kitti360_256x256_map/model_iou_0.62.pt',
                          image_left_topic=rospy.get_param('~image_left_topic',
                                                           '/camera/left/image_raw'),
                                                           # '/kitti/camera_color_left/image_raw'),
                          image_right_topic=rospy.get_param('~image_right_topic',
                                                            '/camera/right/image_raw'),
                                                            # '/kitti/camera_color_right/image_raw'),
                          cam_left_info_topic=rospy.get_param('~cam_left_info_topic',
                                                              '/kitti/camera_color_left/camera_info'),
                          cam_right_info_topic=rospy.get_param('~cam_right_info_topic',
                                                               '/kitti/camera_color_right/camera_info'),
                          odom_topic='/orb_slam3/odom/',
                          )
    time.sleep(1.)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        proc.run()
        rate.sleep()
