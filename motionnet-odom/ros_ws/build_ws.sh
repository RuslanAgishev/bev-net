#!/bin/bash

ROS_DISTRO=melodic
source /opt/ros/$ROS_DISTRO/setup.bash

if [[ $(uname -m) == "x86_64" ]]
then
	echo "x86 Arch"
    # catkin_make -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
	   #          -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m \
	   #          -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so \
	   #          -DCMAKE_BUILD_TYPE=Release
	catkin_make -DPYTHON_EXECUTABLE=/home/ruslan/miniconda3/envs/dl/bin/python3 \
	            -DPYTHON_INCLUDE_DIR=/home/ruslan/miniconda3/envs/dl/include/python3.6m \
	            -DPYTHON_LIBRARY=/home/ruslan/miniconda3/envs/dl/lib/libpython3.6m.so \
	            -DCMAKE_BUILD_TYPE=Release
else
	echo "ARM Arch"
    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 \
                -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
                -DPYTHON_LIBRARY=/usr/lib/$(uname -m)-linux-gnu/libpython3.6m.so \
                -DCMAKE_BUILD_TYPE=Release
fi

source devel/setup.bash
