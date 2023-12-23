#!/bin/bash
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -P datasets
unzip datasets/Replica.zip -d datasets
cp datasets/replica.json datasets/Replica/camera_parameters.json