#!/bin/bash
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz -P datasets/TUM-RGBD
tar -xvzf datasets/TUM-RGBD/rgbd_dataset_freiburg1_desk.tgz -C datasets/TUM-RGBD
wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz -P datasets/TUM-RGBD
tar -xvzf datasets/TUM-RGBD/rgbd_dataset_freiburg2_xyz.tgz -C datasets/TUM-RGBD
wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz -P datasets/TUM-RGBD
tar -xvzf datasets/TUM-RGBD/rgbd_dataset_freiburg3_long_office_household.tgz -C datasets/TUM-RGBD
cp datasets/tum_rgbd.json datasets/TUM-RGBD/camera_parameters.json