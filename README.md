

# Triton MPPI Simulation Branch 

This package depends on ROS Noetic and the TurtleBot3 Gazebo simulation.

## Prerequisites

### Install Required ROS Packages

Run the following command to install the necessary ROS packages:

```bash
sudo apt install ros-noetic-navigation ros-noetic-turtlebot3-gazebo
```

### Additional Dependencies

1. Ensure the `amcl` package is included in your ROS workspace.
2. Follow the [Stingray Camera README](https://gitlab.com/HCRLab/stingray-robotics/stingray_camera/-/tree/main) for download and setup instructions.

## How to Run the Package  

1. Create a ROS workspace and a `src` folder:
    ```bash
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/src
    ```

2. Clone the repositories into the `src` folder:
    ```bash
    git clone <repository_url>
    ```

3. Build the workspace using `catkin_make`:
    ```bash
    cd ~/catkin_ws
    catkin_make
    ```

4. Launch the simulation environment:
    ```bash
    roslaunch triton_mppi triton_navigation.launch
    ```

5. Start the controller node:
    ```bash
    rosrun triton_mppi main.py
    ```

6. Use RViz to set the initial pose and goal:
    - Open RViz and ensure the `2D Pose Estimate` and `2D Nav Goal` tools are enabled.
    - Click on the map to set the robot's initial pose using `2D Pose Estimate`.
    - Set the goal position by clicking on the map with the `2D Nav Goal` tool.
