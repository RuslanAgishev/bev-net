#!/bin/bash

# install packages
apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# setup keys
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
LANG=C.UTF-8
LC_ALL=C.UTF-8

ROS_DISTRO=melodic
rosdep init && rosdep update --rosdistro $ROS_DISTRO

# install ROS packages
apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-desktop \
    ros-$ROS_DISTRO-jsk-recognition-msgs \
    ros-$ROS_DISTRO-usb-cam \
    && rm -rf /var/lib/apt/lists/*

# Install ROS dependencies
source /opt/ros/${ROS_DISTRO}/setup.bash

