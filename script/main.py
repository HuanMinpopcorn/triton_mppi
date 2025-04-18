
#!/usr/bin/env python3

import numpy as np
from MPPI_Controller_CPU_ROS import MPPI_Controller
import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist, PoseStamped
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import tf.transformations

class SimpleModel:
    def __init__(self, model_name='triton'):
        """
        Initialize the SimpleModel class

        Args:
            model_name (str): Name of the model in Gazebo
        """
        # Initialize ROS publisher for cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Initialize subscriber for goal poses from RViz
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal2', PoseStamped, self.goal_callback)

        # Default goal state [x, y, theta]
        self.goal = np.array([0.0, 0.0, 0.0])
        self.new_goal_received = False

        # Initialize service client for get_model_state
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Model name in Gazebo
        self.model_name = model_name

        # Create GetModelStateRequest object
        self.model_state_req = GetModelStateRequest()
        self.model_state_req.model_name = self.model_name
        self.model_state_req.relative_entity_name = 'world'

        # Initialize Twist message for publishing
        self.twist_msg = Twist()

        

    def goal_callback(self, msg):
        """
        Callback function for goal pose messages from RViz

        Args:
            msg (PoseStamped): Goal pose message
        """
        # Extract position
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Extract orientation (quaternion) and convert to euler angles
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        # Convert quaternion to euler angles
        euler = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])
        theta = euler[2]  # yaw angle

        # Update goal
        self.goal = np.array([x, y, theta])
        self.new_goal_received = True

        rospy.loginfo(f"New goal received: x={x}, y={y}, theta={theta}")

    def get_current_state(self):
        """
        Get the current state of the robot from Gazebo

        Returns:
            numpy.ndarray: Current state [x, y, theta]
        """
        try:
            # Call the service to get model state
            result = self.get_state_service(self.model_state_req)

            # Extract position
            x = result.pose.position.x
            y = result.pose.position.y

            # Extract orientation (quaternion) and convert to euler angles
            qx = result.pose.orientation.x
            qy = result.pose.orientation.y
            qz = result.pose.orientation.z
            qw = result.pose.orientation.w

            # Convert quaternion to euler angles
            euler = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])
            theta = euler[2]  # yaw angle

            return np.array([x, y, theta])

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return np.array([0.0, 0.0, 0.0])

    def step(self, control, dt):
        """
        Execute control and update state

        Args:
            state: Current state [x, y, theta]
            control: Control inputs [v, omega]
            dt: Time step

        Returns:
            numpy.ndarray: Updated state after applying control
        """
        # Extract control inputs
        v_cmd, omega_cmd = control

        # Create and publish Twist message
        self.twist_msg.linear.x = v_cmd
        self.twist_msg.linear.y = 0.0
        self.twist_msg.linear.z = 0.0
        self.twist_msg.angular.x = 0.0
        self.twist_msg.angular.y = 0.0
        self.twist_msg.angular.z = omega_cmd

        # Publish control command
        self.cmd_vel_pub.publish(self.twist_msg)

        # Wait for the control to take effect
        rospy.sleep(dt)


def main():
    # Initialize ROS node
    rospy.init_node('mppi_test', anonymous=True)

    # MPPI parameters
    K = 1000  # number of samples
    N = 15    # time horizon
    num_opt = 1  # number of optimization iterations
    dt = 0.1   # timestep
    T = 5 # total time

    # Control limits covariance
    Sigma_c = np.diag([0.5, 0.3])  # variance for [v, omega]

    # Cost parameters
    nu = 0.95  # exploration variance
    lambda_ = 0.1  # temperature
    R = np.diag([0.1, 0.1])  # control cost matrix

    # Initialize model
    model = SimpleModel()

    # Wait for a goal from RViz
    rospy.loginfo("Waiting for a goal pose from RViz...")
    rospy.loginfo("Use the '2D Nav Goal' button in RViz to set a goal.")

    # Initialize MPPI controller with default goal
    controller = MPPI_Controller(
        K=K,
        N=N,
        num_opt=num_opt,
        dt=dt,
        Sigma_c=Sigma_c,
        nu=nu,
        lambda_=lambda_,
        R=R,
        goal=model.goal
    )

    # Get initial state from Gazebo
    initial_state = model.get_current_state()
    current_state = initial_state.copy()
    current_state_log = np.zeros((3, int(T/dt)))
    current_state_log[:, 0] = current_state

    # Main control loop
    rate = rospy.Rate(1/dt)  # Control loop rate
    i = 0
    running = False

    try:
        while not rospy.is_shutdown():
            # Check if a new goal has been received
            if model.new_goal_received:
                # Update controller with new goal
                controller = MPPI_Controller(
                    K=K,
                    N=N,
                    num_opt=num_opt,
                    dt=dt,
                    Sigma_c=Sigma_c,
                    nu=nu,
                    lambda_=lambda_,
                    R=R,
                    goal=model.goal,
                )

                # Reset state log
                i = 0
                current_state = model.get_current_state()
                current_state_log = np.zeros((3, int(T/dt)))
                current_state_log[:, 0] = current_state
                initial_state = current_state.copy()

                # Start running
                running = True
                model.new_goal_received = False
                rospy.loginfo("Starting navigation to goal...")

            # If we're running and haven't reached the maximum time steps
            if running and i < int(T/dt):
                # Get current state
                current_state = model.get_current_state()
                current_state_log[:, i] = current_state

                # Set initial state for controller
                controller.set_initial_state(current_state)

                # Compute optimal control
                optimal_control_sequence = controller.optimize()

                # Execute control
                model.step(optimal_control_sequence[:, 0], dt)

                # Shift control sequence for next iteration
                controller.shift_ctrl_sequence()

                # Print progress
                print(f"Time step: {i}, Current state: {current_state}, Goal: {model.goal}")

                # Check if we've reached the goal
                dist_to_goal = np.sqrt((current_state[0] - model.goal[0])**2 +
                                      (current_state[1] - model.goal[1])**2 +
                                      (current_state[2] - model.goal[2])**2)

                if dist_to_goal < 0.2:  # 20cm threshold
                    rospy.loginfo("Goal reached!")
                    running = False

                    # Stop the robot
                    stop_msg = Twist()
                    model.cmd_vel_pub.publish(stop_msg)

                    # Plot the trajectory
                    # plt.figure(figsize=(10, 8))
                    # plt.plot(current_state_log[0, :i+1], current_state_log[1, :i+1], 'b-', label='Real Trajectory')
                    # plt.plot(initial_state[0], initial_state[1], 'go', markersize=10, label='Initial State')
                    # plt.plot(model.goal[0], model.goal[1], 'ro', markersize=10, label='Goal State')
                    # plt.xlabel('X')
                    # plt.ylabel('Y')
                    # plt.title('MPPI Optimal Trajectory')
                    # plt.legend()
                    # plt.grid(True)
                    # plt.axis('equal')
                    # plt.show()

                i += 1
            elif running and i >= int(T/dt):
                # We've reached the maximum time steps
                rospy.loginfo("Maximum time steps reached without reaching the goal.")
                running = False

                # Stop the robot
                stop_msg = Twist()
                model.cmd_vel_pub.publish(stop_msg)

                # Plot the trajectory
                plt.figure(figsize=(10, 8))
                plt.plot(current_state_log[0, :], current_state_log[1, :], 'b-', label='Real Trajectory')
                plt.plot(initial_state[0], initial_state[1], 'go', markersize=10, label='Initial State')
                plt.plot(model.goal[0], model.goal[1], 'ro', markersize=10, label='Goal State')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('MPPI Optimal Trajectory')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                plt.show()

            # Sleep to maintain the control loop rate
            rate.sleep()

    except KeyboardInterrupt:
        # Stop the robot when the program is interrupted
        stop_msg = Twist()
        model.cmd_vel_pub.publish(stop_msg)
        rospy.loginfo("Program interrupted, stopping the robot.")

    finally:
        # Make sure the robot stops when the program ends
        stop_msg = Twist()
        model.cmd_vel_pub.publish(stop_msg)
        rospy.loginfo("Program finished, stopping the robot.")

if __name__ == "__main__":
    main()
