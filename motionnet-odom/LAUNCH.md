## ROS
- Launch [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM)
  SLAM to provide odometry information (here is an example with Velodyne HDL-64 lidar used in KITTI):

  ```bash
  cd /path/to/A-LOAM/docker
  ./run.sh 64
  ```

- Launch MotionNet node:
  ```bash
  roslaunch motionnet_ros motionnet_odom.launch
  ```

- Play a bag file, for example:
  ```bash
  rosbag play --clock kitti_2011_09_29_drive_0071_synced.bag /kitti/velo/pointcloud:=/velodyne_points -r 0.2
  ```