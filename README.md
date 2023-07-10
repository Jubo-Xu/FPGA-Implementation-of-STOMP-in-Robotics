## CPU Version
For Benchmark purposes a CPU version of STOMP has been implemented. This code is base on the ROS_industrial [STOMP repository](https://github.com/ros-industrial/stomp/tree/main). The STOMP code is integrated with in a ROS2 package which contain a gazebo simulation for testing purposes. 

## run the code
This section aim to explain how to run the ros2 package. This section assume that you have install all the prerequisite.

1. create a ROS2 workspace 
```
mkdir ROS2_Workspace
```
3. enter the workspace
```
cd ROS2_Workspace
```
4. Clone the reporsitary: 
```
git clone [url of the repo]
```
5. As always make sure to source ros2_humble : 
```
. ~/ros2_humble/install/local_setup.bash
```
6. In order to build the package type there is no need for some make command as colcon do it for you: 
```
colcon build --symlink
```
7.source the package
```
source install/setup.bash
```
8. We will now launch the simu, the world command ensure to load the world specify by the path. In here the world shoudl contain one sphere obstacle.
```
ros2 launch my_bot launch_sim.launch.py world:= my_bot/worlds/empty.world
```
9. We will now make the robotic arm move. first change directory
```
cd build/my_bot
```
9. Then execute the executable manually
```
.\joint_command
```

you should see the robot move. 


