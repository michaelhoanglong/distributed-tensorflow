#!/bin/bash
# set up tensorflow
sudo apt-get update
sudo apt-get install -y python3-pip 
sudo apt-get install -y python3-dev
pip3 install awscli
pip3 install tensorflow

# run training
git clone https://github.com/michaelhoanglong/distributed-tensorflow.git /home/ubuntu/tensorflow