"""General use functions.
"""
import os
import cv2
import time
import argparse
import numpy as np
from scipy.optimize import nnls


class GateNavigator:
    def __init__(self, gates_positions, threshold=0.28):
        """
        Initialize the navigator with the positions of all gates.
        
        Parameters:
        - gates_positions: A list of gate positions, where each position is np.array([x, y, z]).
        """
        self.gates_positions = gates_positions
        self.current_gate_index = 0  # Start with the first gate as the target
        self.reached_gate = False
        self.previous_position = None
        self.dot_threshold = 1e-4
        self.threshold=threshold
        self.completed_laps = 0
    
    def update_drone_position(self, drone_position):
        """
        Update the drone's position and determine if it's time to target the next gate.
        
        Parameters:
        - drone_position: The current position of the drone as np.array([x, y, z]).
        """
        # if self.current_gate_index >= len(self.gates_positions):
        #     print("All gates have been passed.")
        #     return
        
        if self.previous_position is not None:
            movement_vector = drone_position - self.previous_position

            # Calculate the distance to the current target gate
            gate_position = self.gates_positions[self.current_gate_index]
            # gate_position = np.array([current_gate.position.x_val, current_gate.position.y_val, current_gate.position.z_val])

            distance_to_gate = np.linalg.norm(drone_position - gate_position)

            # Calculate the vector from the previous position to the gate position
            gate_vector = gate_position - self.previous_position
            
            # Calculate the dot product of the movement vector and the gate vector
            dot_product = np.dot(movement_vector, gate_vector)

            # print ("dot: ", dot_product)
            # print (f"distance: {distance_to_gate:.4f} dot: {dot_product:.4f}")

            # Check if the drone has reached the current gate
            if distance_to_gate < self.threshold and dot_product < self.dot_threshold:
                self.reached_gate = True
                # print (f"distance: {distance_to_gate:.4f} dot: {dot_product:.4f}")
        
        # Update the previous position for the next iteration
        self.previous_position = drone_position

        # If the drone has reached the gate and is now moving away, switch to the next gate
        if self.reached_gate: #and distance_to_gate > threshold:
            # print (self.current_gate_index, ". Gate has been passed!")
            self.current_gate_index += 1  # Move to the next gate

            if self.current_gate_index == len(self.gates_positions):
                self.current_gate_index = 0
                self.completed_laps += 1

            self.reached_gate = False  # Reset the reached_gate flag
            # if self.current_gate_index < len(self.gates_positions):
            #     print(f"Switched to gate {self.current_gate_index}.")
            # else:
            #     print("All gates have been passed.")

    
    def get_current_target_gate(self):
        """
        Get the position of the current target gate.
        
        Returns:
        - The position of the current target gate as np.array([x, y, z]), or None if all gates are passed.
        """
        if self.current_gate_index < len(self.gates_positions):
            return self.gates_positions[self.current_gate_index]
        else:
            return None

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


def record_video(image_folder):
    # Get a list of PNG image filenames in the directory
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Sort the image filenames to ensure they are in the correct order
    # image_files.sort()
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Specify the output video file name and codec
    output_file = image_folder + '/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the dimensions of the first image
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object
    video_writer = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    # Iterate over the image files and write them to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()