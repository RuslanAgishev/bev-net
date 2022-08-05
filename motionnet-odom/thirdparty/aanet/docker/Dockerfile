FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

# Install AANet dependencies
RUN pip install \
    scikit-image \
    imageio
# Build deformable convolutions
COPY ./aanet/nets/deform_conv /tmp/aanet/nets/deform_conv
RUN cd /tmp/aanet/nets/deform_conv && bash build.sh

# Install ROS
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*
# setup keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO kinetic
# bootstrap rosdep
RUN rosdep init \
    && rosdep update
# install ros packages
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-desktop \
    ros-$ROS_DISTRO-ros-numpy \
    ros-$ROS_DISTRO-catkin python-catkin-tools \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-tf && \
    rm -rf /var/lib/apt/lists/*

# setup ROS workspace
RUN pip install rospkg \
                empy \
                catkin-pkg \
                opencv-python \
                cupy-cuda100
RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python3.so
COPY ./aanet/ros_ws/src /tmp/ros_ws/src
SHELL ["/bin/bash","-c"]
RUN source /opt/ros/$ROS_DISTRO/setup.bash && apt-get update && \
    cd /tmp/ros_ws && catkin_make -DPYTHON_EXECUTABLE=/opt/conda/bin/python3.6m \
                                   -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m \
                                   -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.6m.so

WORKDIR /root/aanet
