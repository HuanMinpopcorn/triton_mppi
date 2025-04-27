#!/usr/bin/env python3

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

def load_trajectory_data(file_path):
    """
    Load trajectory data from a pickle file
    
    Args:
        file_path (str): Path to the pickle file
    
    Returns:
        dict: Trajectory data dictionary
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_map(map_path):
    """
    Load map from a PGM file
    
    Args:
        map_path (str): Path to the PGM file
    
    Returns:
        numpy.ndarray: Map as a numpy array
    """
    try:
        # Load the PGM file
        img = Image.open(map_path)
        # Convert to numpy array
        map_data = np.array(img)
        # Invert the map so obstacles are dark and free space is light
        # map_data = 255 - map_data
        return map_data
    except Exception as e:
        print(f"Error loading map: {e}")
        return None

def create_costmap_colormap():
    """
    Create a custom colormap for ROS costmaps
    
    Returns:
        ListedColormap: Custom colormap for costmap visualization
    """
    # Define the color mapping
    # Index corresponds to cost value (0-255)
    colors = np.zeros((256, 4))  # RGBA

    # Free space (0): black
    colors[0] = [0.0, 0.0, 0.0, 1.0]

    # Unknown space (255): gray
    colors[255] = [0.5, 0.5, 0.5, 1.0]

    # Lethal obstacle (254): purple
    colors[254] = [0.5, 0.0, 0.5, 1.0]

    # Inscribed inflated obstacle (253): cyan
    colors[253] = [0.0, 1.0, 1.0, 1.0]

    # Inflated obstacles (1-252): gradient from blue to red
    for i in range(1, 253):
        ratio = (i - 1) / 251  # Normalize between 0 and 1
        colors[i] = [ratio, 0.0, 1.0 - ratio, 1.0]  # RGB gradient

    # Create the colormap
    return ListedColormap(colors)

def plot_trajectory_stages(data, map_data=None, map_origin=(-10, -10), map_resolution=0.05, title=None):
    """
    Plot the state trajectory divided into four stages with local costmaps
    
    Args:
        data (dict): Trajectory data dictionary
        map_data (numpy.ndarray, optional): Map data
        map_origin (tuple, optional): Map origin (x, y) in meters
        map_resolution (float, optional): Map resolution in meters/pixel
        title (str, optional): Plot title
    """
    # Extract states from the data
    states = np.array(data['states'])
    goals = np.array(data['goals'])
    costmaps = data['costmaps']
    
    # Create custom colormap for costmaps
    costmap_cmap = create_costmap_colormap()
    
    # Create four separate figures
    figs = []
    axs = []
    
    # Figure 1: Complete trajectory
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    figs.append(fig1)
    axs.append(ax1)
    
    # Figures 2-4: Different stages
    for i in range(3):
        fig, ax = plt.subplots(figsize=(10, 8))
        figs.append(fig)
        axs.append(ax)
    
    # Calculate indices for the four stages
    num_steps = len(states)
    indices = [
        num_steps - 1,  # Complete trajectory (all steps)
        num_steps // 3,  # First third
        2 * num_steps // 3,  # Second third
        num_steps - 1  # End
    ]
    
    for i, idx in enumerate(indices):
        ax = axs[i]
        
        # Plot the map if available
        if map_data is not None:
            # Calculate extent for the map
            height, width = map_data.shape
            extent = [
                map_origin[0], 
                map_origin[0] + width * map_resolution,
                map_origin[1], 
                map_origin[1] + height * map_resolution
            ]
            ax.imshow(map_data, cmap='gray', origin='upper', extent=extent, alpha=0.7)
        
        # For the first figure, plot the complete trajectory
        if i == 0:
            ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=1, label='Triton Trajectory')
        else:
            # For other figures, plot trajectory up to the specific stage
            ax.plot(states[:idx+1, 0], states[:idx+1, 1], 'b-', linewidth=2, label='Triton Trajectory')
        
        # Plot the start point with orientation arrow
        start_x, start_y, start_theta = states[0]
        ax.plot(start_x, start_y, 'go', markersize=6, label='Start')
        # Add arrow to show orientation at start
        arrow_length = 0.3
        dx = arrow_length * np.cos(start_theta)
        dy = arrow_length * np.sin(start_theta)
        ax.arrow(start_x, start_y, dx, dy, head_width=0.1, head_length=0.15, fc='g', ec='g')
        
        # Plot the current point with orientation arrow
        current_x, current_y, current_theta = states[idx]
        ax.plot(current_x, current_y, 'ro', markersize=6, label='Current Position')
        # Add arrow to show current orientation
        dx = arrow_length * np.cos(current_theta)
        dy = arrow_length * np.sin(current_theta)
        ax.arrow(current_x, current_y, dx, dy, head_width=0.1, head_length=0.15, fc='r', ec='r')
        
        # Plot the goal with orientation arrow
        goal_x, goal_y = goals[0, 0], goals[0, 1]
        goal_theta = 0.0  # Assuming goal orientation is not specified, use 0
        if goals.shape[1] > 2:  # If goal orientation is available
            goal_theta = goals[0, 2]
        ax.plot(goal_x, goal_y, 'mx', markersize=10, label='Goal')
        # Add arrow to show goal orientation
        dx = arrow_length * np.cos(goal_theta)
        dy = arrow_length * np.sin(goal_theta)
        ax.arrow(goal_x, goal_y, dx, dy, head_width=0.1, head_length=0.15, fc='m', ec='m')
        
        # Plot the local costmap if available
        if i > 0 and costmaps[idx] is not None:  # Skip costmap for the complete trajectory figure
            costmap = costmaps[idx]
            costmap_data = costmap['data']
            
            # Get costmap parameters
            resolution = costmap['resolution']
            width = costmap['width']
            height = costmap['height']
            origin_x = costmap['origin_x']
            origin_y = costmap['origin_y']
            
            # Calculate extent for the costmap
            costmap_extent = [
                origin_x,
                origin_x + width * resolution,
                origin_y,
                origin_y + height * resolution
            ]
            
            # Plot the costmap with custom colormap
            costmap_img = ax.imshow(
                costmap_data.reshape(height, width),
                cmap=costmap_cmap,
                origin='lower',
                extent=costmap_extent,
                alpha=0.7,
                vmin=0,
                vmax=255,
                label='Local Costmap'
            )
            
            # Add colorbar for the costmap
            # cbar = fig.colorbar(costmap_img, ax=ax, fraction=0.046, pad=0.04)
            # cbar.set_label('Cost')
            
            # Add a proxy artist for the costmap to include in the legend
            costmap_patch = patches.Patch(color=[0.3, 0.0, 0.7, 0.7], label='Local Costmap')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(costmap_patch)
            ax.legend(handles=handles, loc='upper left')
        else:
            # Add grid and legend for plots without costmap
            ax.legend(loc='upper left')
        
        # Add labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        if i == 0:
            ax.set_title(f'Triton Trajectory')
        else:
            # Calculate relative time from start
            relative_time = idx * data.get('dt', 0.1)
            ax.set_title(f'Snapshot at t={relative_time:.2f}s')
        
        # Add grid
        ax.grid(True)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
    
        # Add overall title to each figure
        # if title:
        #     figs[i].suptitle(f"{title} - {stage_names[i]}", fontsize=16)
        # else:
        #     figs[i].suptitle(f'Robot Trajectory - {stage_names[i]}', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    return figs, axs

def save_trajectory_plots(figs, output_dir, base_filename):
    """
    Save all trajectory plots to the specified directory
    
    Args:
        figs (list): List of matplotlib figure objects
        output_dir (str): Directory to save the plots
        base_filename (str): Base filename for the plots
    
    Returns:
        list: List of saved file paths
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage names for the filenames
    stage_names = ['complete', 'first_third', 'second_third', 'final']
    
    saved_files = []
    
    # Save each figure
    for i, fig in enumerate(figs):
        # Create filename
        filename = f"{base_filename}_{stage_names[i]}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save the figure
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        saved_files.append(filepath)
        print(f"Saved plot to: {filepath}")
    
    return saved_files

def main():
    # Find the results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    parent_dir = os.path.dirname(script_dir)
    print(parent_dir)
    results_dir = os.path.join(parent_dir, "results")
    
    # Path to the map file
    map_path = os.path.join(parent_dir, "maps", "map.pgm")
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all pickle files in the results directory
    pickle_file = glob(os.path.join(results_dir, "trajectory_data_without_amcl_1.pkl"))
    
    if not pickle_file:
        print(f"No pickle files found in {results_dir}")
        return
    
    # Load the most recent file by default
    data_file = pickle_file[0]
    print(f"Loading the trajectory data: {os.path.basename(data_file)}")
    
    # Load the data
    data = load_trajectory_data(data_file)
    
    # Load the map
    map_data = None
    if os.path.exists(map_path):
        print(f"Loading map from: {map_path}")
        map_data = load_map(map_path)
    else:
        print(f"Map file not found: {map_path}")
    
    # Plot the trajectory stages with the map and local costmaps
    figs, axs = plot_trajectory_stages(
        data, 
        map_data=map_data, 
        map_origin=(-10, -10),  # Adjust based on your map's origin
        map_resolution=0.05,    # Adjust based on your map's resolution
        title=f"Trajectory from {os.path.basename(data_file)}"
    )
    
    # Ask user if they want to save plots
    save_plots = input("Do you want to save the plots? (y/n): ").lower().strip() == 'y'
    
    if save_plots:
        # Create plots directory inside results directory
        plots_dir = os.path.join(results_dir, "plots_without_amcl")
        
        # Save the plots
        base_filename = os.path.splitext(os.path.basename(latest_file))[0]
        save_trajectory_plots(figs, plots_dir, base_filename)
        print("Plots saved successfully.")
    else:
        print("Plots not saved.")
    
    # Show the plots
    plt.show()
    
    print("Available data keys:", list(data.keys()))
    print(f"Trajectory length: {len(data['states'])} timesteps")
    
    # Calculate distance to goal
    final_state = np.array(data['states'][-1])
    goal = np.array(data['goals'][0])  # Assuming goal doesn't change
    distance_to_goal = np.sqrt((final_state[0] - goal[0])**2 + (final_state[1] - goal[1])**2)
    
    print(f"Final state: {final_state}")
    print(f"Goal state: {goal}")
    print(f"Final distance to goal: {distance_to_goal:.3f} meters")

if __name__ == "__main__":
    main()
