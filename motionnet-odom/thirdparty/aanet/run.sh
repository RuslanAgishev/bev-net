#!/bin/bash
trap : SIGTERM SIGINT

roscore &
ROSCORE_PID=$!
sleep 1

rviz -d ./ros_ws/src/aanet_ros/rviz/aanet_config.rviz &
RVIZ_PID=$!

docker run \
    --runtime=nvidia \
    -it \
    --rm \
    --net=host \
    -v $(pwd):/root/aanet \
    ros:aanet \
    /bin/bash -c \
        "source /opt/ros/kinetic/setup.bash; \
        source /tmp/ros_ws/devel/setup.bash;
        roslaunch aanet_ros aanet.launch"

wait $ROSCORE_PID
wait $RVIZ_PID

if [[ $? -gt 128 ]]
then
    kill $ROSCORE_PID
    kill $RVIZ_PID
fi
