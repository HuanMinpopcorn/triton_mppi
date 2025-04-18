
"""
@brief: Python implementation of MPPI Controller (CPU Version)
@authors: Adapted from MATLAB implementation
"""

import numpy as np
import rospy
import warnings
import sys
import tf
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, PoseStamped

# Import visualization messages here to avoid dependency issues
from visualization_msgs.msg import Marker, MarkerArray

class MPPI_Controller:
    def __init__(self, K, N, num_opt, dt, Sigma_c, nu, lambda_, R, goal, occupancy_grid=None, costmap_topic='/move_base/local_costmap/costmap'):
        """
        Initialize MPPI Controller

        Args:
            K (int): Number of samples
            N (int): Time horizon
            num_opt (int): Number of optimization iterations
            dt (float): Timestep
            Sigma_c (numpy.ndarray): Covariance matrix for control noise
            nu (float): Exploration variance
            lambda_ (float): Temperature parameter
            R (numpy.ndarray): Control cost matrix
            goal (numpy.ndarray): Goal state for vehicle
            occupancy_grid (numpy.ndarray, optional): Occupancy grid for obstacle detection
            costmap_topic (str): ROS topic for costmap subscription
        """
        # MPPI settings
        self.K = K
        self.N = N
        self.optimization_loop_num = num_opt
        self.dt = dt
        self.nu = nu
        self.lambda_ = lambda_
        self.R = R
        self.goal = goal
        self.occupancy_grid = occupancy_grid

        # Controller dimensions
        self.n = 3  # number of states
        self.ctrl_dim = 2  # dimension of control space

        # Control sequence and state storage
        self.ctrl_sequence = np.zeros((self.ctrl_dim, self.N))
        self.optimized_ctrl_sequence = np.zeros((self.ctrl_dim, self.N))
        self.rollout_states = np.zeros((self.n, self.N, self.K))
        self.rollout_costs = np.zeros(self.K)

        # Adjust covariance with importance sampling
        self.Sigma_c = nu * Sigma_c

        # Cost weights
        self.weight_goal_stage = 1.0
        self.weight_goal_terminate = 1.0
        self.obstacle_weight = 100.0  # Weight for obstacle avoidance

        # Costmap variables using double buffering approach
        self.costmap = None  # Will hold the entire costmap as a single dictionary
        self.costmap_received = False

        # Initialize TF listener for coordinate transformations
        self.tf_listener = tf.TransformListener()

        # Subscribe to costmap topic
        self.costmap_sub = rospy.Subscriber(costmap_topic, OccupancyGrid, self.costmap_callback)

    def costmap_callback(self, msg):
        """
        Callback function for costmap subscription

        Args:
            msg (OccupancyGrid): Costmap message
        """
        # Extract costmap metadata
        resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Extract orientation of the costmap
        qx = msg.info.origin.orientation.x
        qy = msg.info.origin.orientation.y
        qz = msg.info.origin.orientation.z
        qw = msg.info.origin.orientation.w

        # Convert 1D array to 2D numpy array for easier access
        # Note: In OccupancyGrid, -1 (255) = unknown, 0 = free, 100 = occupied
        costmap_data = np.array(msg.data).reshape(height, width)

        # Create binary occupancy grid where True = occupied (cost >= 50)
        occupancy_grid = costmap_data >= 50  # Consider cells with cost >= 50 as obstacles

        # Create a new costmap object with all the data
        new_costmap = {
            'resolution': resolution,
            'width': width,
            'height': height,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'orientation': [qx, qy, qz, qw],
            'frame_id': msg.header.frame_id,
            'data': costmap_data,
            'occupancy_grid': occupancy_grid
        }

        # Atomically replace the costmap reference
        self.costmap = new_costmap

        # Mark that we've received a costmap
        self.costmap_received = True

        rospy.logdebug(f"Received costmap: {width}x{height} cells, resolution: {resolution}m, frame: {msg.header.frame_id}")

    def system_dynamics(self, state, control):
        """
        Compute system dynamics for 3D dubin's car

        Args:
            state: Current state [x, y, theta]
            control: Control inputs [v, omega]
        Returns:
            next_state: Next state after applying control
        """
        x, y, theta = state
        v_cmd, omega_cmd = control

        next_state = np.array([
            x + v_cmd * np.cos(theta) * self.dt,
            y + v_cmd * np.sin(theta) * self.dt,
            theta + omega_cmd * self.dt
        ])
        return next_state

    def check_collision_with_footprint(self, x, y, theta):
        """
        Check if the robot's footprint is in collision with obstacles

        Args:
            x (float): Robot's x position in world coordinates
            y (float): Robot's y position in world coordinates
            theta (float): Robot's orientation in world coordinates

        Returns:
            bool: True if in collision, False otherwise
        """
        if not self.costmap_received or self.costmap is None:
            return False

        # Get a reference to the current costmap (atomic operation)
        costmap = self.costmap

        try:
            # # Get the transform from world frame to costmap frame
            # costmap_frame = costmap.get('frame_id', 'map')  # Default to 'map' if not specified
            costmap_frame = 'base_link'
            world_frame = 'world'  # Or whatever frame the robot position is in

            # Create a PoseStamped for the robot position in world frame
            robot_pose = PoseStamped()
            robot_pose.header.frame_id = world_frame
            robot_pose.header.stamp = rospy.Time(0)
            robot_pose.pose.position.x = x
            robot_pose.pose.position.y = y
            robot_pose.pose.position.z = 0.0

            # Convert theta to quaternion
            q = tf.transformations.quaternion_from_euler(0, 0, theta)
            robot_pose.pose.orientation.x = q[0]
            robot_pose.pose.orientation.y = q[1]
            robot_pose.pose.orientation.z = q[2]
            robot_pose.pose.orientation.w = q[3]

            # Wait for the transform to be available
            if self.tf_listener.waitForTransform(costmap_frame, world_frame, rospy.Time(0), rospy.Duration(0.1)):
                # Transform the robot position to costmap frame
                robot_pose_costmap = self.tf_listener.transformPose(costmap_frame, robot_pose)

                # Extract the position in costmap frame
                map_x = robot_pose_costmap.pose.position.x
                map_y = robot_pose_costmap.pose.position.y

                # Define robot footprint as a circle
                robot_radius = 0.2  # meters - adjust based on your robot's size

                # Convert map coordinates to grid cell indices
                center_x = int(map_x / costmap['resolution'])
                center_y = int(map_y / costmap['resolution'])

                # Calculate how many cells to check based on robot radius
                cells_to_check = int(robot_radius / costmap['resolution'])

                # Check if the robot center is far outside the map bounds
                if (center_x < -cells_to_check or center_x >= costmap['width'] + cells_to_check or
                    center_y < -cells_to_check or center_y >= costmap['height'] + cells_to_check):
                    # Robot is completely outside the map
                    return False  # Alternatively, return True if you want to be conservative

                # Check all cells within the robot footprint
                for dx in range(-cells_to_check, cells_to_check + 1):
                    for dy in range(-cells_to_check, cells_to_check + 1):
                        # Skip cells outside the circular footprint
                        if dx*dx + dy*dy > cells_to_check*cells_to_check:
                            continue

                        check_x = center_x + dx
                        check_y = center_y + dy

                        # Check if within bounds and is obstacle
                        if (0 <= check_x < costmap['width'] and
                            0 <= check_y < costmap['height'] and
                            costmap['occupancy_grid'][check_y, check_x]):
                            return True  # Collision detected
            else:
                # If transform is not available, use a simpler approach
                rospy.logwarn(f"Transform from {world_frame} to {costmap_frame} not available. Using simple transformation.")
                

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error in collision checking: {e}.")
    

        return False  # No collision


    def state_cost(self, state, is_terminal=False):
        """
        Compute state cost

        Args:
            state: Current state
            is_terminal: Whether this is terminal state
        Returns:
            cost: Cost value
        """
        cost = 0.0

        # Extract vehicle position
        x, y, theta = state[:3]

        # Vehicle to goal cost
        x_goal = x - self.goal[0]
        y_goal = y - self.goal[1]
        theta_goal = theta - self.goal[2]

        # Normalize angle difference
        theta_goal = np.arctan2(np.sin(theta_goal), np.cos(theta_goal))

        # Goal distance
        goal_dist = np.sqrt(x_goal**2 + y_goal**2)

        if is_terminal:
            # Terminal cost
            cost += self.weight_goal_terminate * (goal_dist + np.abs(theta_goal))
        else:
            # Running cost
            cost += self.weight_goal_stage * (goal_dist + np.abs(theta_goal))

        # Obstacle cost if we've received a costmap
        if self.costmap_received:
            # Check for collision using the robot's footprint
            # No lock needed with double buffering
            if self.check_collision_with_footprint(x, y, theta):
                cost += self.obstacle_weight

        return cost

    def control_cost(self, u, du):
        """
        Compute control cost

        Args:
            u: Control input
            du: Control noise
        Returns:
            cost: Control cost value
        """
        control_cost = 0
        control_cost += 0.5 * u.T @ self.R @ u + u.T @ self.R @ du + 0.5 * (1 - 1/self.nu) * du.T @ self.R @ du
        return control_cost

    def rollouts(self):
        """
        Perform rollouts using CPU

        Returns:
            costs: Array of costs for each rollout
            noise: Generated noise for controls
        """
        # Generate random noise
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.ctrl_dim),
            cov=self.Sigma_c,
            size=(self.K, self.N)
        ).transpose(0, 2, 1)  # Reshape to (K, ctrl_dim, N)

        # Initialize cost array
        costs = np.zeros(self.K)

        # Perform rollouts
        for k in range(self.K):
            state = self.ini_state.copy()

            # Initialize trajectory cost
            traj_cost = 0.0

            # Simulate trajectory
            for n in range(self.N):
                # Get control input with noise
                u = self.ctrl_sequence[:, n]
                du = noise[k, :, n]
                u_noisy = u + du

                # Store state
                self.rollout_states[:, n, k] = state

                # Update state
                state = self.system_dynamics(state, u_noisy)

                # Compute running cost
                traj_cost += self.state_cost(state) + self.control_cost(u, du)

            # Add terminal cost
            traj_cost += self.state_cost(state, is_terminal=True)
            costs[k] = traj_cost

        return costs, noise

    def compute_control_update(self, costs, noise):
        """
        Compute control update based on rollouts

        Args:
            costs: Array of costs for each rollout
            noise: Generated noise for controls
        Returns:
            updated_sequence: Updated control sequence
        """
        beta = np.min(costs)
        exp_costs = np.exp(-1.0 / self.lambda_ * (costs - beta))
        normalizer = np.sum(exp_costs) + 1e-10

        weighted_noise = np.zeros_like(self.ctrl_sequence)
        for k in range(self.K):
            weight = exp_costs[k] / normalizer
            weighted_noise += weight * noise[k]

        updated_sequence = self.ctrl_sequence + weighted_noise
        return updated_sequence

    def optimize(self):
        """
        Main function to run MPPI optimization
        """
        for _ in range(self.optimization_loop_num):
            # Perform rollouts
            costs, noise = self.rollouts()

            # Store costs for analysis
            self.rollout_costs = costs

            # Compute control update
            updated_sequence = self.compute_control_update(costs, noise)

            # Update control sequence
            self.ctrl_sequence = updated_sequence

            # Save optimized control sequence
            self.optimized_ctrl_sequence = self.ctrl_sequence.copy()

        # # Visualize the costmap and trajectory (if available)
        # try:
        #     self.visualize_costmap_and_trajectory()
        # except Exception as e:
        #     rospy.logwarn(f"Visualization failed: {e}")

        return self.ctrl_sequence

    def set_initial_state(self, state):
        """Set initial state for optimization"""
        self.ini_state = state.copy()

    def get_optimal_trajectory(self):
        """Get the optimal trajectory"""
        # Compute optimal trajectory using optimized controls
        self.optimized_states = np.zeros((self.n, self.N))
        state = self.ini_state.copy()
        self.optimized_states[:, 0] = state

        for t in range(1, self.N):
            state = self.system_dynamics(state, self.optimized_ctrl_sequence[:, t-1])
            self.optimized_states[:, t] = state

        return self.optimized_states

    def shift_ctrl_sequence(self):
        """Shift control sequence forward in time"""
        self.ctrl_sequence[:, :-1] = self.ctrl_sequence[:, 1:]
        self.ctrl_sequence[:, -1] = 0  # Zero out the last control

    def optimal_cost_function(self):
        """Compute the cost of the optimal trajectory"""
        optimal_cost = 0
        optimal_traj = self.get_optimal_trajectory()


        for i in range(self.N - 1):
            ctrl = self.optimized_ctrl_sequence[:, i]
            ctrl_cost = ctrl.T @ self.R @ ctrl
            state_cost = self.state_cost(optimal_traj[:, i])
            optimal_cost += (state_cost + ctrl_cost) * self.dt

        # Add terminal cost
        optimal_cost += self.state_cost(optimal_traj[:, -1], is_terminal=True)

        return optimal_cost

    # def visualize_costmap_and_trajectory(self):
    #     """
    #     Visualize the costmap and planned trajectory using ROS markers
    #     """
    #     if not self.costmap_received or self.costmap is None:
    #         rospy.logwarn("Cannot visualize costmap: No costmap data received yet")
    #         return

    #     try:
    #         # Create publishers if they don't exist yet
    #         if not hasattr(self, 'costmap_pub'):
    #             self.costmap_pub = rospy.Publisher('/mppi/costmap_visualization', MarkerArray, queue_size=1, latch=True)

    #         if not hasattr(self, 'trajectory_pub'):
    #             self.trajectory_pub = rospy.Publisher('/mppi/trajectory', Marker, queue_size=1)

    #         # Get a reference to the current costmap (atomic operation)
    #         costmap = self.costmap

    #         # Visualize costmap as cube markers
    #         marker_array = MarkerArray()
    #         marker_id = 0

    #         # Create a marker for occupied cells
    #         obstacle_marker = Marker()
    #         obstacle_marker.header.frame_id = costmap['frame_id']  # Use the costmap's frame
    #         obstacle_marker.header.stamp = rospy.Time.now()
    #         obstacle_marker.ns = "costmap"
    #         obstacle_marker.id = marker_id
    #         obstacle_marker.type = Marker.CUBE_LIST
    #         obstacle_marker.action = Marker.ADD
    #         obstacle_marker.scale.x = costmap['resolution']
    #         obstacle_marker.scale.y = costmap['resolution']
    #         obstacle_marker.scale.z = 0.1  # Height of the cubes
    #         obstacle_marker.color.r = 1.0
    #         obstacle_marker.color.g = 0.0
    #         obstacle_marker.color.b = 0.0
    #         obstacle_marker.color.a = 0.7
    #         obstacle_marker.pose.orientation.w = 1.0

    #         # Add points for occupied cells
    #         for y in range(costmap['height']):
    #             for x in range(costmap['width']):
    #                 if costmap['occupancy_grid'][y, x]:
    #                     p = Point()
    #                     p.x = costmap['origin_x'] + (x + 0.5) * costmap['resolution']
    #                     p.y = costmap['origin_y'] + (y + 0.5) * costmap['resolution']
    #                     p.z = 0.05  # Slightly above ground
    #                     obstacle_marker.points.append(p)

    #         marker_array.markers.append(obstacle_marker)
    #         self.costmap_pub.publish(marker_array)

    #         # Visualize planned trajectory
    #         trajectory = self.get_optimal_trajectory()

    #         traj_marker = Marker()
    #         traj_marker.header.frame_id = costmap['frame_id']  # Use the costmap's frame
    #         traj_marker.header.stamp = rospy.Time.now()
    #         traj_marker.ns = "trajectory"
    #         traj_marker.id = 0
    #         traj_marker.type = Marker.LINE_STRIP
    #         traj_marker.action = Marker.ADD
    #         traj_marker.scale.x = 0.05  # Line width
    #         traj_marker.color.g = 1.0
    #         traj_marker.color.a = 1.0
    #         traj_marker.pose.orientation.w = 1.0

    #         # Add points for trajectory
    #         for i in range(trajectory.shape[1]):
    #             p = Point()
    #             p.x = trajectory[0, i]
    #             p.y = trajectory[1, i]
    #             p.z = 0.1  # Slightly above ground
    #             traj_marker.points.append(p)

    #         self.trajectory_pub.publish(traj_marker)

    #     except Exception as e:
    #         rospy.logerr(f"Error visualizing costmap: {e}")
