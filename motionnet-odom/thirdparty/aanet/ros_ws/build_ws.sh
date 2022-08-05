#! /bin/bash
catkin_make -DPYTHON_EXECUTABLE=/opt/conda/bin/python3.6m \
            -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m \
            -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.6m.so
