### Real Time Person Orientation Estimation using Colored Pointclouds ###
![Repo_Eyecatcher](/classifier/misc/Repo_eyecatcher.png?raw=true "Repo_Eyecatcher")

#### Installation and setup ####
This package has been tested on Ubuntu Trusty 16.04 LTS (64-Bit) and Mint 18.3 (Sylvia) both with ROS Kinetic.

The following steps describe the installation procedure on a clean ubuntu 16.04.

Install Reguirements:

`sudo apt-get install git libopencv-dev`

Install ROS like from http://wiki.ros.org/kinetic/Installation/Ubuntu:

`sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'`

`sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116'`

`sudo apt-get update`

`sudo apt-get install ros-kinetic-desktop-full`

`sudo rosdep init`

`rosdep update`

`echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc`

`source ~/.bashrc`

Create a new catkin workspace:

`mkdir -p ~/catkin_ws/src`

`cd ~/catkin_ws/`

`catkin_make`

Clone the repo:

`git clone https://github.com/TimWengefeld/pointcloud_person_orientation_estimation.git`

`cd ..`

Build the code:

`catkin_make -DCMAKE_BUILD_TYPE=RelWithDebInfo`

`source devel/setup.bash`
