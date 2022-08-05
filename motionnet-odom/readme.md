# BEV-Net

The goal of this project is constructing a local map for mobile robot navigation from input sensory data.


## Relevant work

For SOTA on bird-eye-view (BEV) maps construction,
please, refer to [BEV-notes in notion](https://www.notion.so/66d056f8ec984a4d8c179fbc232fac71?v=37b9a0485cd54d9f88fad8c2670f7af9).

This project is build based on [MotionNet](https://github.com/pxiangwu/MotionNet).
The MotionNet documentation and installation instructions are available at [motionnet.md](https://gitlab.com/vedu/bev-net/-/blob/motionnet-odom/motionnet.md).


## MotionNet with Odometry

We include odometry estimation to MotionNet input to compensate for ego-motion.
SOTA open-source visual and lidar-based odometry estimation is
collected at [Odometry-notes in notion](https://www.notion.so/f5b0d91a080a47648cf871fdfa32455d?v=9acb1fc08b57486e81f1bcd9b867cac5).

Several ROS-wrappers for the odometry sources are included in [thirdparty](https://gitlab.com/vedu/bev-net/-/tree/motionnet-odom/thirdparty) folder. Please refer for individual packages installation instructions to use odometry of your choice.
For example, [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) documantation is included as [README.md](https://gitlab.com/vedu/bev-net/-/blob/motionnet-odom/thirdparty/A-LOAM/README.md).


## ROS

MotionNet ROS-node is wrapped as [motionnet_ros](https://gitlab.com/vedu/bev-net/-/tree/motionnet-odom/ros_ws/src/motionnet_ros)
package.

Building the package:

```bash
cd ./ros_ws/
./build_ws.sh
source devel/setup.bash
```

### MotionNet with A-LOAM

This section assumes, that you installed MotionNet dependencies and built the ROS package.

Build docker image and run A-LOAM ROS node:

```bash
cd thirdparty/A-LOAM/docker
make build
./run.sh 64
```

Launch MotionNet node as follows:
```bash
roslaunch motionnet_ros motionnet_odom.launch
```

Provide sensors data at `/odom` and `/velodyne_points` topics. For example, play a -bag file:

```bash
rosbag play --clock dundich_v4.1.bag
```

### MotionNet with ORB-SLAM2

Install ORB-SLAM2 dependencies, as described in its documentation, [README.md](https://gitlab.com/vedu/bev-net/-/blob/motionnet-odom/thirdparty/ORB_SLAM2/README.md).

Specify your local `PATH` to [ORB_SLAM2](https://gitlab.com/vedu/bev-net/-/tree/motionnet-odom/thirdparty/ORB_SLAM2) package.

```bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
```

Run stereo example:
```bash
roscore
rosrun ORB_SLAM2 Stereo Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml false
```

Ones you installed MotionNet dependencies and built the ROS package, launch:
```bash
roslaunch motionnet_ros motionnet_odom.launch odom_topic:=/orb_slam2/odom
```

Provide stereo images data at `/camera/left/image_raw` and `/camera/right/image_raw` topics. For example, play a -bag file:

```bash
rosbag play --clock kitti_2011_09_29_drive_0071_synced.bag /kitti/camera_color_left/image_raw:=/camera/left/image_raw /kitti/camera_color_right/image_raw:=/camera/right/image_raw 
```
