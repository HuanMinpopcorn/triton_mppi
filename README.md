

### Dependencies

1. Install the required ROS package:
   - turtlebot3_gazebo for simulator setup
   
    ```bash
    cd ~/catkin_ws/src/
    git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

    sudo apt install ros-melodic-turtlebot3-gazebo
    sudo apt install ros-melodic-navigation
    ```

3. Ensure `amcl` is included in your ROS workspace.

4. Follow the [Stingray Camera README](https://gitlab.com/HCRLab/stingray-robotics/stingray_camera/-/tree/main?ref_type=heads) for download and setup instructions.
