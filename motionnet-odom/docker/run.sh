#!/bin/bash
trap : SIGTERM SIGINT

function abspath() {
    # generate absolute path from relative path
    # $1     : relative filename
    # return : absolute path
    if [ -d "$1" ]; then
        # dir
        (cd "$1"; pwd)
    elif [ -f "$1" ]; then
        # file
        if [[ $1 = /* ]]; then
            echo "$1"
        elif [[ $1 == */* ]]; then
            echo "$(cd "${1%/*}"; pwd)/${1##*/}"
        else
            echo "$(pwd)/$1"
        fi
    fi
}

roscore &
ROSCORE_PID=$!
sleep 1

rviz -d ../ros_ws/src/motionnet_ros/configs/motionnet.rviz &
RVIZ_PID=$!

MOTIONNET_DIR=$(abspath "..")

docker run \
    -it \
    --runtime=nvidia \
    --rm \
    --net=host \
    -v $MOTIONNET_DIR:/root/motionnet \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    ros:motionnet \
    /bin/bash -c \
        "source /opt/ros/melodic/setup.bash; \
        cd /root/motionnet/ros_ws/; \
        rm -rf devel build; \
        catkin config \
                -DCMAKE_BUILD_TYPE=Release \
                -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
                -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m \
                -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so; \
        catkin build motionnet_ros; \
        source devel/setup.bash;
        export PYTHONPATH=/root/motionnet:$PYTHONPATH; \
        export PYTHONPATH=/root/motionnet/nuscenes-devkit/python-sdk:$PYTHONPATH; \
        roslaunch motionnet_ros motionnet_odom.launch"

wait $ROSCORE_PID
wait $RVIZ_PID

if [[ $? -gt 128 ]]
then
    kill $ROSCORE_PID
    kill $RVIZ_PID
fi