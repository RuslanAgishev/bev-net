# Lift, Splat, Shoot ([LSS](https://gitlab.com/vedu/bev-net/-/blob/master/lift-splat-shoot/LSS.md))

Notes on how to train LSS model on different datasets and run on custom data.

## Preparation

- Nuscenes

    Download nuscenes data from [https://www.nuscenes.org/](https://www.nuscenes.org/). Install dependencies.

    Change `dataroot` path in [./configs/nuscenes_config.yaml](https://gitlab.com/vedu/bev-net/-/blob/master/lift-splat-shoot/configs/nuscenes_config.yaml)

- KITTI-360
    
    Download KITTI360 data from [http://www.cvlibs.net/datasets/kitti-360/index.php](http://www.cvlibs.net/datasets/kitti-360/index.php).

    Change `dataroot` path in [./configs/kitti360_config.yaml](https://gitlab.com/vedu/bev-net/-/blob/master/lift-splat-shoot/configs/kitti360_config.yaml)


## Pre-trained Models
Download a pre-trained BEV road layout and vehicle segmentation models from here:
[https://files.sberdisk.ru/s/FKGFNtoccDXwNgC](https://files.sberdisk.ru/s/FKGFNtoccDXwNgC)

Put pretrained model checkpoints in `./weights` directory.

Results on Nuscenes maps (`32 x 32 m, 0.125 m resolution`) validation set for layout estimation based on single camera input:

| Road IOU      | Vehicle IOU   |
|:-------------:|:-------------:| 
| 83            | 34            |

Model: [weights](https://files.sberdisk.ru/s/nLKb4FcoMT4fD5i),
[config](https://files.sberdisk.ru/s/HdRn5jw7iGbbMBD).

Results on KITTI-360 maps (`41 x 41 m, 0.16 m resolution`) validation set (sequence 2013_05_28_drive_0004_sync):

| Road IOU      |
|:-------------:|
| 62            |

Model: [weights and config](https://files.sberdisk.ru/s/aG8ktnEmPg76pm5).

## Explore Input/Output Data

Take a look at the [./notebooks](https://gitlab.com/vedu/bev-net/-/tree/master/lift-splat-shoot/notebooks)
folder for tutorials how to prepare the data for model training:
- [Nuscenes](https://gitlab.com/vedu/bev-net/-/blob/master/lift-splat-shoot/notebooks/example_mono.ipynb):
    local map prediction from single camera input (image, extrinsics, intrinsics).
- [KITTI360](https://gitlab.com/vedu/bev-net/-/blob/master/lift-splat-shoot/notebooks/explore_kitti360_bev_maps.ipynb):
    local map prediction from stereo-camera input (left and right image, extrinsics, intrinsics).
- [Discriminator](https://gitlab.com/vedu/bev-net/-/blob/master/lift-splat-shoot/notebooks/discriminator.ipynb):
    training LSS model with discriminator for output layout correction, based on Open Street Maps or nuscenes data.

## Semantic Mapping (ROS)

The output semantic mapping is produced from a sequnece of local map predictions and odometry data.
Resultant occupancy grid is created with the help of
[log odds](http://ais.informatik.uni-freiburg.de/teaching/current-ws/mapping/pdf/slam11-gridmaps-4.pdf)
representation.

### Mono input, LSS trained on Nuscenes

<img src="./imgs/lss_semantic_mapping.gif">

The example is launch on nuscenes ROS bag data. One possible option is to use
[nuscenes2bag](https://github.com/clynamen/nuscenes2bag) tool to generate the data in proper format.
Then run:

```
roscore
python nuscenes_semantic_mapping_ros.py
rosbag play /path/to/nuscenes_bags/61.bag -r 0.3
```

### Stareo input, LSS trained on KITTI360

LSS semantic mapping with stereo input. In this demo, on the top are left and right input images,
on the bottom right is the local map prediction by LSS model in the left camera field of view,
on the bottom left is the resultant global map. It is obtained by combining and filtering the sequence
of local map predictions with the help of
[ORB_SLAM3](https://gitlab.com/vedu/bev-net/-/tree/master/motionnet-odom/thirdparty/ORB_SLAM3) odometry.

<img src="./imgs/gt_odom_lss_stereo_mapping.gif">

Execute the following commands to get the result:

```
roscore
rosrun ORB_SLAM3 Stereo Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml false
python kitti360_lss_semantic_mapping_ros.py
rosbag play kitti_2011_09_26_drive_0096_synced.bag \
        /kitti/camera_color_left/image_raw:=/camera/left/image_raw \
        /kitti/camera_color_right/image_raw:=/camera/right/image_raw \
        -r 0.3 --pause
```

Hit Space once the ORB_SLAM is loaded the Vocabulary and LSS node is up.
