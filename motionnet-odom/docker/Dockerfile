FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV CATKIN_WS=/root/motionnet/ros_ws

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libturbojpeg \
    software-properties-common \
    libboost-all-dev \
    libssl-dev \
    libgeos-dev \
    wget \
    nano \
    sudo \
    git \
    python3-matplotlib \
    python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install MotionNet dependencies
RUN pip install \
    pyquaternion \
    matplotlib==3.1.0 \
    pillow==6.2.1 \
    tqdm \
    opencv-python==4.4.0.40 \
    numba==0.48.0 \
    cupy-cuda101 \
    rospkg \
    empy \
    sklearn \
    cachetools==4.0.0 \
    imageio==2.8.0

# Install ROS
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*
# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic
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

# # Build motionnet_ros
# WORKDIR $CATKIN_WS
# ENV TERM xterm
# ENV PYTHONIOENCODING UTF-8
# RUN rm -rf devel build
# RUN catkin config \
#       --extend /opt/ros/$ROS_DISTRO \
#       --cmake-args \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
#       -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m \
#       -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so; \
#     catkin build motionnet_ros && \
#     sed -i '/exec "$@"/i \
#             source "$CATKIN_WS/devel/setup.bash"' /ros_entrypoint.sh

WORKDIR /root/motionnet
