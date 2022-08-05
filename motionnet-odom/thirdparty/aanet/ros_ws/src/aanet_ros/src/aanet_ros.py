#!/usr/bin/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU id
os.environ["OMP_NUM_THREADS"] = "8"  # CPU is working in 1 thread
# ROS libraries
import rospy
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import PointCloud2
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from cv_bridge import CvBridge, CvBridgeError
PATH_TO_AANET = '/root/aanet'
sys.path.append(PATH_TO_AANET)
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from dataloader import transforms
import nets
from utils import utils
import skimage.io
import scipy.misc as ssc
import kitti_util
from generate_lidar import project_disp_to_points, gen_sparse_points
from generate_lidar import xyz_array_to_pointcloud2, xyzrgb_array_to_pointcloud2
from collections import deque


class AANetImgProcessor:
	# constants
	MAX_ALLOWED_DELAY_SEC = float( rospy.get_param("~/aanet_ros/max_allowed_proc_delay", -1) ) # sec
	IMAGENET_MEAN = [0.485, 0.456, 0.406]
	IMAGENET_STD = [0.229, 0.224, 0.225]

	def __init__(self):
		self.bridge = CvBridge()
		self.img_height = 384
		self.img_width = 1248
		self.load_model()
		self.test_transform = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
		self.calib = kitti_util.Calibration(os.path.join(PATH_TO_AANET, 'ros_ws/src/aanet_ros/src/calib.txt'))
		# subscribed Topic
		self.left_image_topic = rospy.get_param("~/aanet_ros/left_img_topic")
		self.right_image_topic = rospy.get_param("~/aanet_ros/right_img_topic")
		self.semseg_image_topic = rospy.get_param("~/aanet_ros/semseg_img_topic")
		self.left_image_msg = None
		self.right_image_msg = None
		self.semseg_msg = None
		self.left_sub = rospy.Subscriber(self.left_image_topic, Image, self.callback_left,  queue_size = 1)
		self.right_sub = rospy.Subscriber(self.right_image_topic, Image, self.callback_right,  queue_size = 1)
		self.semseg_sub = rospy.Subscriber(self.semseg_image_topic, Image, self.callback_semseg,  queue_size = 1)
		print('[INFO] Subscribed to:', self.left_image_topic)
		print('[INFO] Subscribed to:', self.right_image_topic)
		print('[INFO] Subscribed to:', self.semseg_image_topic)
		# disparity and pointcloud publishers
		self.out_disp_topic = rospy.get_param("~/aanet_ros/output_disp_topic")
		self.out_pc_topic = rospy.get_param("~/aanet_ros/output_pc_topic")
		self.disp_pub = rospy.Publisher(self.out_disp_topic, Image, queue_size=1)
		self.pc_pub = rospy.Publisher(self.out_pc_topic, PointCloud2, queue_size=1)
		self.num_lidar_rays = int( rospy.get_param("~/aanet_ros/num_lidar_rays") )
		# relevant in time data processing
		self.received_first_frame = False
		self.t_started_ros_node = rospy.Time.now()
		self.t_shift = None
		self.delays_array = deque(maxlen=2)

	def load_model(self,):
		self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = nets.AANet(max_disp=192,
								num_downsample=2,
								feature_type='aanet',
								no_feature_mdconv=False,
								feature_pyramid=False,
								feature_pyramid_network=True,
								feature_similarity='correlation',
								aggregation_type='adaptive',
								num_scales=3,
								num_fusions=6,
								num_stage_blocks=1,
								num_deform_blocks=3,
								no_intermediate_supervision=True,
								refinement_type='stereodrnet',
								mdconv_dilation=2,
								deformable_groups=2).to(self.device)
		utils.load_pretrained_net(self.model, os.path.join(PATH_TO_AANET, 'pretrained/aanet_kitti12-e20bb24d.pth'), no_strict=True)
		if torch.cuda.device_count() > 1:
			print('=> Use %d GPUs' % torch.cuda.device_count())
			self.model = torch.nn.DataParallel(self.model)
		# Inference mode
		self.model.eval()

	def preprocess_images(self, left_img, right_img):
		"""
		:param left_img: np.array of shape (H, W, 3)
		:param right_img: np.array of shape (H, W, 3)
		:return: left: torch.Tensor of shape (1, 3, self.img_height, self.img_width)
		:return: right: torch.Tensor of shape (1, 3, self.img_height, self.img_width)
		"""
		sample = {}
		sample['left'] = left_img.astype(np.float32)
		sample['right'] = right_img.astype(np.float32)
		sample = self.test_transform(sample)
		left = torch.unsqueeze(sample['left'], 0).to(self.device)
		right = torch.unsqueeze(sample['right'], 0).to(self.device)
		# Padding
		ori_height, ori_width = left_img.shape[:2]
		if ori_height < self.img_height or ori_width < self.img_width:
			top_pad = self.img_height - ori_height
			right_pad = self.img_width - ori_width
			# Pad size: (left_pad, right_pad, top_pad, bottom_pad)
			left = F.pad(left, [0, right_pad, top_pad, 0])
			right = F.pad(right, [0, right_pad, top_pad, 0])
		return left, right

	def postprocessing(self, pred_disp, color_image, left_tensor_size, header=None):
		"""
		:param pred_disp: torch.Tensor of shape (batch_size, self.img_height, self.img_width)
		:return: disp_list: List of len=batch_size, each element of it is of shape (self.img_height, self.img_width)
		"""
		if pred_disp.size(-1) < left_tensor_size(-1):
			pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
			pred_disp = F.interpolate(pred_disp, (left_tensor_size(-2), left_tensor_size(-1)),
									  mode='bilinear') * (left_tensor_size(-1) / pred_disp.size(-1))
			pred_disp = pred_disp.squeeze(1)  # [B, H, W]
		# Crop
		ori_height, ori_width = color_image.shape[:2]
		if ori_height < self.img_height or ori_width < self.img_width:
			top_pad = self.img_height - ori_height
			right_pad = self.img_width - ori_width
			if right_pad != 0:
				pred_disp = pred_disp[:, top_pad:, :-right_pad]
			else:
				pred_disp = pred_disp[:, top_pad:]

		for b in range(pred_disp.size(0)):
			disp = pred_disp[b].detach().cpu().numpy()  # [H, W]
			result = (disp * 256.).astype(np.uint16)
			res_msg = self.bridge.cv2_to_imgmsg(result, "16UC1")
			if header is not None: res_msg.header = header
			self.disp_pub.publish(res_msg) # publish disparity
			# convert disparity to pointcloud
			pc, valid_pc_indexes = project_disp_to_points(self.calib, disp.astype(np.uint16), max_high=1.)
			# create color map for pc visualization
			# colors = color_image.reshape(-1, 3)
			# colors = colors[valid_pc_indexes]
			# pc, colors = gen_sparse_points(pc, colors, H=self.num_lidar_rays)
			# pc_xyz = pc[:, :3]
			# colors = colors[:, :3]
			#
			# if pc_xyz.shape[0] == colors.shape[0]:
			# 	print('publishing colored pointcloud')
			# 	pc_msg = xyzrgb_array_to_pointcloud2(pc_xyz, colors,
			# 										 stamp=header.stamp,
			# 										 frame_id="velo_link")
			# else:
			# 	pc_msg = xyz_array_to_pointcloud2(pc_xyz,
			# 									  stamp=header.stamp,
			# 									  frame_id="velo_link")
			pc, colors = gen_sparse_points(pc, H=self.num_lidar_rays)
			pc_msg = xyz_array_to_pointcloud2(pc[:, :3],
											  stamp=header.stamp,
											  frame_id="velo_link")
			self.pc_pub.publish(pc_msg) # publish pointcloud

	def callback_left(self, img_msg):
		try:
			self.left_image_msg = img_msg
		except CvBridgeError as e:
			print(e)

	def callback_right(self, img_msg):
		try:
			self.right_image_msg = img_msg
		except CvBridgeError as e:
			print(e)

	def callback_semseg(self, img_msg):
		try:
			self.semseg_msg = img_msg
		except CvBridgeError as e:
			print(e)

	def get_delay(self, msg):
		"""
		Calculates delay between current time moment and last msg received.
		:return: delay_sec: float, [sec]
		"""
		if not self.received_first_frame:
			t0 = msg.header.stamp  # first frame time
			self.t_shift = self.t_started_ros_node - t0  # time shift between ROS node start and first frame received
			self.received_first_frame = True
			self.delays_array.append(0.0)

		t_current = rospy.Time.now()  # current time
		t_msg = msg.header.stamp  # last msg received time
		tmp = (t_current - (t_msg + self.t_shift)).to_sec()
		self.delays_array.append(tmp)
		delay_sec = self.delays_array[1] - self.delays_array[0]  # tmp
		return np.abs(delay_sec)

	def run(self):
		# Convert ROS msgs to cv2_images
		if 'compressed' in self.left_image_topic:
			left_img = self.bridge.compressed_imgmsg_to_cv2(self.left_image_msg, "bgr8")
		else:
			left_img = self.bridge.imgmsg_to_cv2(self.left_image_msg, "bgr8")
		if 'compressed' in self.right_image_topic:
			right_img = self.bridge.compressed_imgmsg_to_cv2(self.right_image_msg, "bgr8")
		else:
			right_img = self.bridge.imgmsg_to_cv2(self.right_image_msg, "bgr8")
		if self.semseg_msg is not None:
			if 'compressed' in self.semseg_image_topic:
				semseg_img = self.bridge.compressed_imgmsg_to_cv2(self.semseg_msg, "bgr8")
			else:
				semseg_img = self.bridge.imgmsg_to_cv2(self.semseg_msg, "bgr8")
			colors = semseg_img[...,(2,1,0)]
		else:
			colors = left_img[...,(2,1,0)]

		t0 = time.perf_counter()

		# Preprocessing
		time_start = time.perf_counter()
		left, right = self.preprocess_images(left_img, right_img)
		print(f'Input processing took {1000 * (time.perf_counter() - time_start):.1f} [ms]')

		# Inference
		# left = torch.rand(size=(1, 3, 960, 3130)).to(self.device)
		# right = torch.rand(size=(1, 3, 960, 3130)).to(self.device)
		time_start = time.perf_counter()
		with torch.no_grad():
			pred_disp = self.model(left, right)[-1]  # [B, H, W]
		print(f'Inference of shape {left.size()} took {1000 * (time.perf_counter() - time_start):.1f} [ms]')

		# Postprocessing
		time_start = time.perf_counter()
		self.postprocessing(pred_disp, colors, left.size, self.left_image_msg.header)
		print(f'Output processing took {1000 * (time.perf_counter() - time_start):.1f} [ms]')

		print(f'\nOveral run time: {1000 * (time.perf_counter() - t0):.1f} [ms]\n')


def main():
	# Initializes ROS node
	rospy.init_node('aanet_node', anonymous=True)
	proc = AANetImgProcessor()

	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		if proc.left_image_msg is not None and proc.right_image_msg is not None:
			# dt = (proc.left_image_msg.header.stamp - proc.right_image_msg.header.stamp).to_sec()
			# print(f"\nLeft and right images delay: {1000*dt:.1f} [ms]\n")
			proc.run()
		rate.sleep()


if __name__ == '__main__':
	main()
