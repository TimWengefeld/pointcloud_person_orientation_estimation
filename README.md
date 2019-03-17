## Real Time Person Orientation Estimation using Colored Pointclouds ##
![Repo_Eyecatcher](/classifier/misc/Repo_eyecatcher.png?raw=true "Repo_Eyecatcher")

### Introduction ###

This repository contains a ROS-based C++ / Python implementation of the classification and regression approach to estimate the upper body orientation of a person described in the paper:
> *Real Time Person Orientation Estimation using Colored Pointcloud*  

This approach is based on the repo from `https://github.com/spencer-project/spencer_human_attribute_recognition` described in the papers:

> *Real-Time Full-Body Human Attribute Classification in RGB-D Using a Tessellation Boosting Approach*  
> by Timm Linder and Kai O. Arras   
> IEEE/RSJ Int. Conference on Intelligent Robots and Systems (IROS), Hamburg, Germany, 2015.

and

> *Tracking People in 3D Using a Bottom-Up Top-Down Detector*  
> by Luciano Spinello, Matthias Luber, Kai O. Arras    
> IEEE International Conference on Robotics and Automation (ICRA'11), Shanghai, China, 2011.  

The original attribute estimation is still included and running, but we made several changes on the code, most notably the following:
* Ported the code to more recent OS and ROS versions
* removed the (not used) HOG based attribute estimation since it won't build on a the new OS version and we don't care about it
* Fixed code and enabled C++11 support for convenience
* Added a node to pre-calculate features for every point cloud file (will speed up the training and testing of different model parameters)

##### TODO: #####
* Port the code to OpenCV 3 (the project will link against the OpenCV 2 from the system while ROS is based on the OpenCV 3, from my experiences with other middleware frameworks this might causes problems in larger applications)
* re-add  the HOG based attribute estimation (maybe) 

##### Attention: #####
Please keep in mind that I am not a ROS expert since I am typically developing with the robotic Middleware MIRA `http://www.mira-project.org/`.
So please expect this code as a pretty example which was never tested in a larger application.

### Installation and setup ###
This package has been tested on Ubuntu Trusty 16.04 LTS (64-Bit) and Mint 18.3 (Sylvia) both with ROS Kinetic.

The following steps describe the installation procedure on a clean ubuntu 16.04.

Install Reguirements:
    
    sudo apt-get install git libopencv-dev

Install ROS like from http://wiki.ros.org/kinetic/Installation/Ubuntu:

    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list`
    sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
    sudo apt-get update
    sudo apt-get install ros-kinetic-desktop-full
    sudo rosdep init
    rosdep update
    echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
    source ~/.bashrc

Create a new catkin workspace:

    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/
    catkin_make

Clone the repo:

    cd src
    git clone https://github.com/TimWengefeld/pointcloud_person_orientation_estimation.git`
    cd ..

Build the code:

    catkin_make -DCMAKE_BUILD_TYPE=RelWithDebInfo
    source devel/setup.bash

### Try the examples ###

By executing `sh src/pointcloud_person_orientation_estimation/classifier/scripts/orientation_example.sh` you will receive some outputs like: 
