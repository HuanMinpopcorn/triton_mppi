

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

5.1 (Option 1: without AMCL) Start the controller node:
    ```bash
    rosrun triton_mppi main.py
    ```

5.2 (Option 2: with AMCL) Start the controller node:
    ```bash
    rosrun triton_mppi main_with_amcl.py
    ```

> **Note:** After executing either option 5.1 or 5.2, you will see a terminal prompt indicating that the Controller is waiting for a navigation goal to be set.


6. Use RViz to set the initial pose and goal:
    - Open RViz and ensure the `2D Pose Estimate` and `2D Nav Goal` tools are enabled.
    - Click on the map to set the robot's initial pose using `2D Pose Estimate`.
    - Set the goal position by clicking on the map with the `2D Nav Goal` tool.

7. Trajectory Data Storage

> **Note:** Once the robot reaches its goal, trajectory data is automatically saved in the `./results/` directory with timestamp information.

8.1 (Option 1: without AMCL) Analyze trajectory results
    ```bash
    rosrun triton_mppi results_without_amcl.py
    ```

8.2 (Option 2: with AMCL) Analyze trajectory results
    ```bash
    rosrun triton_mppi results_with_amcl.py
    ```

> **Note:** After executing either option 8.1 or 8.2, you will be prompted to save the visualization plots. If you choose to save them, the plots will be stored in the `./results/plots_without_amcl` or `./results/plots_with_amcl` directory, respectively.