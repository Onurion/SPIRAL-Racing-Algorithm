import os
import re
import sys
import pickle
import numpy as np
import pybullet as p
import gymnasium
from gymnasium import spaces
from collections import deque
import time
from PIL import Image
import pybullet_data
import pkg_resources
from datetime import datetime
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from envs.BaseDrone_SelfPlay import BaseDrone_SelfPlay


from utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from control.DSLPIDControl import DSLPIDControl
from scipy.interpolate import RegularGridInterpolator

gate_folder = "assets/gate.urdf"

class GateNavigator:
    def __init__(self, gates_positions, threshold=0.28, waypoint_distance=0.5, safe_distance=0.5):
        """
        Initialize the navigator with the positions of all gates.

        Parameters:
        - gates_positions: A list of gate positions, where each position is np.array([x, y, z]).
        - threshold: Distance threshold to consider the gate as passed.
        - waypoint_distance: Desired distance between waypoints.
        - safe_distance: Minimum safe distance from other drones to avoid collisions.
        """
        self.gates_positions = gates_positions
        self.current_gate_index = 0  # Start with the first gate as the target
        self.reached_gate = False
        self.previous_position = None
        self.dot_threshold = 1e-4
        self.threshold = threshold
        self.completed_laps = 0
        self.waypoints = []
        self.waypoint_distance = waypoint_distance
        self.safe_distance = safe_distance
        self.drone_position = None
        self.max_speed = 1.0
        self.position_buffer = []
        self.buffer_size = 5  # Number of positions to average for smoothing

    def update_drone_position(self, drone_position, other_drones_positions, other_drones_velocities):
        """
        Update the drone's position, generate waypoints if needed, and adjust them to avoid collisions.

        Parameters:
        - drone_position: The current position of the drone as np.array([x, y, z]).
        - other_drones_positions: A list of positions of other drones.
        """
        self.drone_position = np.array(drone_position)

        # Check if the drone has passed the current gate
        self._check_gate_passage()

        # Generate waypoints if needed
        if not self.waypoints:
            self._generate_waypoints()

        # Adjust waypoints to avoid collisions
        self._adjust_waypoints(other_drones_positions, other_drones_velocities)

        self.previous_position = self.drone_position

    def _check_gate_passage(self):
        """
        Check if the drone has passed through the current gate.
        """
        gate_position = self.gates_positions[self.current_gate_index]
        distance_to_gate = np.linalg.norm(self.drone_position - gate_position)

        if self.previous_position is not None:
            movement_vector = self.drone_position - self.previous_position
            gate_vector = gate_position - self.previous_position
            dot_product = np.dot(movement_vector, gate_vector)
            # print ("distance_to_gate: ", distance_to_gate, " dot_product: ", dot_product)

            if distance_to_gate < self.threshold and dot_product < self.dot_threshold:
                self.reached_gate = True

        if self.reached_gate:
            self.current_gate_index += 1  # Move to the next gate

            if self.current_gate_index == len(self.gates_positions):
                self.current_gate_index = 0
                self.completed_laps += 1

            self.reached_gate = False
            self.waypoints = []  # Clear waypoints for the next gate

    def _generate_waypoints(self):
        """
        Generate waypoints from the drone's current position to the current gate.
        """
        gate_position = self.gates_positions[self.current_gate_index]
        to_gate = gate_position - self.drone_position
        distance_to_gate = np.linalg.norm(to_gate)
        num_waypoints = max(1, int(distance_to_gate / self.waypoint_distance))
        direction = to_gate / distance_to_gate if distance_to_gate > 0 else np.zeros(3)
        self.waypoints = [self.drone_position + direction * self.waypoint_distance * i for i in range(1, num_waypoints + 1)]
        self.waypoints.append(gate_position)  # Ensure we reach the gate


    def _adjust_waypoints(self, other_drones_positions, other_drones_velocities=None):
        """
        Adjust waypoints to avoid collisions with other drones.

        Parameters:
        - other_drones_positions: A list of positions of other drones.
        """
        adjusted_waypoints = []
        for waypoint in self.waypoints:
            adjusted_waypoint = waypoint.copy()
            for other_pos in other_drones_positions:
                distance = np.linalg.norm(other_pos - adjusted_waypoint)
                if distance < self.safe_distance:
                    # Adjust waypoint away from other drone
                    avoidance_vector = adjusted_waypoint - other_pos
                    if np.linalg.norm(avoidance_vector) > 0:
                        avoidance_vector /= np.linalg.norm(avoidance_vector)
                        adjustment = avoidance_vector * (self.safe_distance - distance)
                        adjusted_waypoint += adjustment
            adjusted_waypoints.append(adjusted_waypoint)
        self.waypoints = adjusted_waypoints

    # def _adjust_waypoints(self, other_drones_positions, other_drones_velocities):
    #     """
    #     Adjust waypoints to avoid future collisions.
    #     """
    #     adjusted_waypoints = []
    #     for idx, waypoint in enumerate(self.waypoints):
    #         adjusted_waypoint = waypoint.copy()
    #         time_to_waypoint = idx * self.waypoint_distance / self.max_speed  # Estimated time to reach waypoint
    #         for other_pos, other_vel in zip(other_drones_positions, other_drones_velocities):
    #             # Predict other drone's future position
    #             predicted_other_pos = other_pos + other_vel * time_to_waypoint
    #             distance = np.linalg.norm(predicted_other_pos - adjusted_waypoint)
    #             if distance < self.safe_distance:
    #                 # Adjust waypoint away from predicted position
    #                 avoidance_vector = adjusted_waypoint - predicted_other_pos
    #                 if np.linalg.norm(avoidance_vector) > 0:
    #                     avoidance_vector /= np.linalg.norm(avoidance_vector)
    #                     adjustment = avoidance_vector * (self.safe_distance - distance)
    #                     adjusted_waypoint += adjustment
    #         adjusted_waypoints.append(adjusted_waypoint)
    #     self.waypoints = adjusted_waypoints

    # def get_next_waypoint(self):
    #     """
    #     Get the next waypoint for the drone to move towards.

    #     Returns:
    #     - The next waypoint as np.array([x, y, z]), or None if no waypoints are available.
    #     """
    #     if self.waypoints:
    #         return self.waypoints.pop(0)
    #     else:
    #         # If no waypoints, set the gate position as the next target
    #         return self.gates_positions[self.current_gate_index]

    def get_next_waypoint(self):
        # Existing code to get the next waypoint...
        if self.waypoints:
            raw_waypoint = self.waypoints.pop(0)

            # Add to position buffer
            self.position_buffer.append(raw_waypoint)
            if len(self.position_buffer) > self.buffer_size:
                self.position_buffer.pop(0)

            # Compute smoothed waypoint
            smoothed_waypoint = np.mean(self.position_buffer, axis=0)
            return smoothed_waypoint
        else:
            return self.gates_positions[self.current_gate_index]

class MultiGates_SEIBR(BaseDrone_SelfPlay):
    def __init__(self, 
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 num_dumb_drones: int=1,
                 num_max_drones: int=4,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 480,
                 ctrl_freq: int = 480,
                 gui=False,
                 record=False,
                 record_folder=None,
                 max_timesteps = 2000,
                 track: int=0,
                 grid_resolution=5, dt=0.1):
        
        self.num_drones = num_drones
        self.dt = dt
        self.drone_states = {}
        self.trajectories = {}
        self.grid_resolution = grid_resolution
        original_init_pos = [0,0,0]
        

        self.penalty_value = 0.1

        self.timesteps = 0
        self.action_size = 8
        self.MAX_TIMESTEPS = max_timesteps
        self.EPISODE_LEN_SEC = 8
        #### Create a buffer for the last .5 sec of actions ########
        self.BUFFER_SIZE = 50#int(ctrl_freq//2)
        self.N_buffer_gate = 2
        self.num_closest_drones = 2
        self.action_buffer = dict()
        self.opponent_buffer = dict()
        ####
        vision_attributes = False
        self.record_folder = record_folder
        self.NUM_DUMB_DRONES = num_dumb_drones

        self.position_history = {}  # Store position history for each agent
        self.velocity_history = {}  # Store velocity history for each agent
        self.mean_velocities = {}  # Store mean velocities for each agent
        self.velocity_window = 10  # Number of velocity measurements to keep

        # self.waypoints = []  # List of positional references

        chn = 0.01
        yaw_chn = 0.01

        self.path_colors = [0.5, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0.5]  # R, W, B in a single list
        self.update_frequency = 60  # Update every 60 steps (1 second at 60Hz)
        self.line_width = 3
        self.max_path_length = 1000  # Maximum number of points in the path
        self.path_item_ids = {}

        

        self.action_mapping = {
            0: [chn, 0, 0, 0],
            1: [-chn, 0, 0, 0],
            2: [0, chn, 0, 0],
            3: [0, -chn, 0, 0],
            4: [0, 0, chn, 0],
            5: [0, 0, -chn, 0],
            6: [0, 0, 0, yaw_chn],
            7: [0, 0, 0, -yaw_chn]
        }
        

        if track == 0:

            self.gate_positions = np.array([
                                [-1.0, 0, 1.0],
                                [-1.2, -1.0, 1.0],
                                [-0.5, -1.5, 1.0],
                                [0.0, -0.75, 1]])
        
            self.gate_rpy = np.array([
                        [0, 0, np.pi/4],
                        [0, 0, np.pi/2],
                        [0, 0, 0],
                        [0, 0, np.pi/2]])
            
        elif track == 1:

            self.gate_positions = np.array([
                [-1.5,  0.5,  1.0],  # Gate 1
                [-2.0, -1.7,  1.5],  # Gate 2
                [0.0, -2.5,  1.5],  # Gate 3
                [ 1.7, -2.5,  1.0],  # Gate 4
                [ 3.0, -0.0,  1.5],  # Gate 5
                [ 1.0,  1.5,  0.6]   # Gate 6
            ])

            self.gate_rpy = np.array([
                [0, 0, np.pi/4],    
                [0, np.pi/6,  np.pi/2],    
                [0, -np.pi/4,  np.pi],   
                [0, -np.pi/8,  np.pi],      
                [0, np.pi/8, np.pi/2],  
                [np.pi/4, 0, 0]     
            ])
        elif track == 2 :

            self.gate_positions = np.array([
                                [-1.0, 0, 1.0],
                                [-2.0, -1.0, 1.0],
                                [-0.5, -1.8, 1.0],
                                [1.0, -0.75, 1]])
        
            self.gate_rpy = np.array([
                        [0, 0, np.pi/4],
                        [0, 0, np.pi/2],
                        [0, 0, 0],
                        [0, 0, np.pi/2]])
                    

        assert len(self.gate_positions) == len( self.gate_rpy)

        self.N_GATES = len(self.gate_positions)
        
        self.gate_quats = [p.getQuaternionFromEuler(euler) for euler in self.gate_rpy]
        # self.navigators = [GateNavigator(self.gate_positions) for i in range(num_drones)]
        


        self.GATE_IDS = []

        dumb_agents = [f"dumb_drone_{n}" for n in range(num_dumb_drones)]
        agents = [f"drone_{n}" for n in range(num_drones)]
        self.num_max_drones = num_max_drones


        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         num_dumb_drones=num_dumb_drones,
                         agents=agents,
                         dumb_agents=dumb_agents,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         record_folder=record_folder,
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        

        self.initial_positions = initialize_drones(self.all_agents, original_init_pos)

        self.navigators = {
            agent: GateNavigator(
                self.gate_positions,
                threshold=0.28,          # As before
                waypoint_distance=0.5,   # Adjust as needed
                safe_distance=1.0        # Adjust as needed
            ) for agent in self.all_agents
        }

        

        #### Create integrated controllers #########################
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.ctrl = {agent: DSLPIDControl(drone_model=DroneModel.CF2X) for agent in self.all_agents}

        #### Set a limit on the maximum target speed ###############
        # if act == ActionType.VEL:
        self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
        self.time_interval = 1 / self.PYB_FREQ  # Time between steps in seconds


    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """


        for i in range(len(self.gate_positions)):
            self.GATE_IDS.append(p.loadURDF(gate_folder,
                                basePosition=self.gate_positions[i],
                                baseOrientation=self.gate_quats[i],
                                useFixedBase=True,
                                physicsClientId=self.CLIENT))
            



    def reset(self):
        original_init_pos = [0,0,0]
        self.GATE_IDS = []
        self.current_action = []
        self.expert_action = []
        self.timesteps = 0
        self.navigators = {agent: GateNavigator(self.gate_positions) for agent in self.all_agents}
        self.position = None
        self.path_item_ids = {}
        self.color_index = {}

    
        self.info = {agent: {"lap_time": [], "velocity": [], "successful_flight": True} for agent in self.all_agents}
        self.n_completion = {agent: 0 for agent in self.all_agents}
        self.previous_lap_time = {agent: 0 for agent in self.all_agents}

        self.prev_gate_index = dict()
        for agent in self.all_agents:
            self.prev_gate_index[agent] = 0
            self.path_item_ids[agent] = []
            self.color_index[agent] = 0

        # Clear any existing debug items
        for item_ids in self.path_item_ids.values():
            for item_id in item_ids:
                p.removeUserDebugItem(item_id)
            item_ids.clear()

        self.initial_positions = initialize_drones(self.all_agents, original_init_pos)


        self.drone_states = {
            drone_id: {
                'position': pos,
                'velocity': np.zeros(3),
                'acceleration': np.zeros(3),
                'trajectory': None,
                'failed': False
            } for drone_id, pos in self.initial_positions.items()
        }
        self.trajectories = {drone_id: [] for drone_id in self.initial_positions}

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping(self.initial_positions)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()


        return self.get_state()

    def get_state(self):
        return {
            drone_id: {
                'position': state['position'],
                'velocity': state['velocity'],
                'current_gate': self.navigators[drone_id].current_gate_index,
                'gate_position': self.gate_positions[self.navigators[drone_id].current_gate_index]
            } for drone_id, state in self.drone_states.items()
        }

    # def _computeDroneFailed(self, drone_state):
    #     # Check if drone has crashed (e.g., hit the ground or exceeded max speed)
    #     if drone_state['position'][2] <= 0:  # Drone hit the ground
    #         return True
    #     if np.linalg.norm(drone_state['velocity']) > self.max_speed:  # Exceeded max speed
    #         return True
    #     return False
    
    def _computeDroneFailed(self, drone_name):
        distance_threshold = 20

        state = self._getDroneStateVector(drone_name)
        drone_pos = np.array(state[0:3])
        drone_rpy = np.array(state[7:10])

        if np.abs(drone_rpy[0]) > 3*np.pi/4 or np.abs(drone_rpy[1]) > 3*np.pi/4:
            return True
        
        for j in range(len(self.gate_positions)):
            euclidean_dist = np.linalg.norm(drone_pos - self.gate_positions[j])
            if euclidean_dist >= distance_threshold: #if it is too far from the gates
                return True
            
                
        return False

    def check_gate_collision(self, drone_name):
        for i in range(len(self.gate_positions)):
            gate_collision = p.getContactPoints(bodyA=self.DRONE_IDS[drone_name],
                                            bodyB=self.GATE_IDS[i],
                                            physicsClientId=self.CLIENT)
            if gate_collision:
                # print ("collision ", drone_name, " time: ", self.timesteps)
                return True
        
        return False
    

    def _computeReward(self, drone_name):
        state_i = self._getDroneStateVector(drone_name)
        drone_i_pos = np.array(state_i[0:3])
        drone_i_quat = np.array(state_i[3:7])
        drone_i_rot = Rotation.from_quat(drone_i_quat)

        distance_threshold = 0.1
        reward = 0

        for j in range(len(self.gate_positions)):
            collision = p.getContactPoints(bodyA=self.DRONE_IDS[drone_name], bodyB=self.GATE_IDS[j], physicsClientId=self.CLIENT)
            
            if collision:
                reward += -self.penalty_value
                # print (f"{self.timesteps} {drone_name} collision with the gate {j}! reward: {reward}")

        for drone_i in self.all_agents:
            if drone_i == drone_name:
                continue 
            
            state_j = self._getDroneStateVector(drone_name)
            drone_j_pos = np.array(state_j[0:3])
            drone_distance = np.linalg.norm(drone_j_pos - drone_i_pos)

            drone_collision = p.getContactPoints(bodyA=self.DRONE_IDS[drone_i], bodyB=self.DRONE_IDS[drone_name], physicsClientId=self.CLIENT)
            if drone_collision:
                reward += -self.penalty_value
                # print (f"{self.timesteps} {drone_name} collision with the {drone_i}! reward: {reward}")
            # elif drone_distance < distance_threshold:
            #     reward += -coeff* self.penalty_value / 2.0
            #     print (f"{self.timesteps} {drone_name} too close to the {drone_i}! reward: {reward}")


        
        gate_index = self.navigators[drone_name].current_gate_index
        gate_pos = self.gate_positions[gate_index]
        gate_quat = self.gate_quats[gate_index]

        gate_rot = Rotation.from_quat(gate_quat)
        
        relative_rot = gate_rot.inv() * drone_i_rot  # Changed to gate_rot.inv() * drone_i_rot
        relative_euler = relative_rot.as_euler('xyz', degrees=False)


        # Calculate Euclidean distance between drone and gate
        euclidean_dist = np.linalg.norm(drone_i_pos - gate_pos)
        distance_reward = np.clip(0.25 / (euclidean_dist + 0.01), 0, 1)


        reward += distance_reward
        # print (f"Current reward: {reward:.4f} Distance reward: {distance_reward:.4f}")

        # Calculate orientation alignment reward
        alignment_error = np.linalg.norm(relative_euler)
        alignment_reward = np.clip(1.0 - (alignment_error / np.pi), 0, 1)  # Normalize between 0 and 1
        alignment_reward_weight = 0.25  # Adjust this weight as needed

        reward += (alignment_reward_weight*alignment_reward)
        # print (f"Current reward: {reward:.4f} Alignment reward: {alignment_reward_weight*alignment_reward:.4f}")

        # time.sleep(0.01)

        return reward
    

    def _preprocessAction(self, action_list):
        rpm = dict()
        max_position_step =  4.0 * self.SPEED_LIMIT * self.CTRL_TIMESTEP  # Maximum distance drone can travel in one control step


        # print ("action_list: ", action_list)

        for agent in self.all_agents:
            target_pos = action_list[agent]
            state = self._getDroneStateVector(drone_name=agent)
            current_pos = state[0:3]
            current_vel = state[10:13]

            # Compute desired velocity towards the target position
            # direction = target_pos - current_pos
            # distance = np.linalg.norm(direction)
            # if distance > 0:
            #     direction /= distance
            #     desired_speed = min(self.SPEED_LIMIT, distance / self.CTRL_TIMESTEP)
            #     desired_velocity = direction * desired_speed
            # else:
            #     desired_velocity = np.zeros(3)


            # Limit the positional change
            position_step = target_pos - current_pos
            distance = np.linalg.norm(position_step)
            if distance > max_position_step:
                position_step = position_step / distance * max_position_step
                target_pos = current_pos + position_step

            current_vel = state[10:13]
            desired_velocity = position_step / self.CTRL_TIMESTEP  # Desired velocity to reach target


            # Use your existing controller to compute RPMs
            rpm_k, _, _ = self.ctrl[agent].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=current_pos,
                cur_quat=state[3:7],
                cur_vel=current_vel,
                cur_ang_vel=state[13:16],
                target_pos=target_pos,
                target_rpy=self.gate_rpy[self.navigators[agent].current_gate_index],
                target_vel=desired_velocity
            )
            rpm[agent] = rpm_k




        # for agent in self.all_agents:
        #     state = self._getDroneStateVector(drone_name=agent)     
        #     euler = np.array(state[7:10]) 
        #     action_k = action_list[agent]
            
        #     drone_pos = np.array(state[0:3])
        #     drone_vel = state[10:13]

        #     self.navigators[agent].update_drone_position(drone_pos)
        #     gate_index = self.navigators[agent].current_gate_index
                            
        #     gate_pos = self.gate_positions[gate_index]
        #     gate_orient = self.gate_rpy[gate_index]
        #     target_orient = gate_orient

        #     target_pos = action_k[:3]

        #     next_pos = self._calculateNextStep(
        #         current_position=state[0:3],
        #         destination=target_pos,
        #         step_size=1,
        #     )


        #     v_unit_vector = target_pos / np.linalg.norm(target_pos)

        #     # print ("self.SPEED_LIMIT * v_unit_vector: ", self.SPEED_LIMIT * v_unit_vector)
            
        #     rpm_k, _, _ = self.ctrl[agent].computeControl(control_timestep=self.CTRL_TIMESTEP,
        #                                             cur_pos=state[0:3],
        #                                             cur_quat=state[3:7],
        #                                             cur_vel=state[10:13],
        #                                             cur_ang_vel=state[13:16],
        #                                             target_pos=next_pos,
        #                                             target_rpy=target_orient,
        #                                             # target_vel=self.SPEED_LIMIT * v_unit_vector # target the desired velocity vector
        #                                             )
        #     rpm[agent] = rpm_k

                    
        return rpm
        

    def step(self):
        rewards = {}
        actions = {}
        done = False

        self.timesteps += 1

        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
                    

        # Gather positions of all drones for collision avoidance
        all_positions = {
            agent: self._getDroneStateVector(drone_name=agent)[0:3]
            for agent in self.all_agents
        }

        all_velocities = {
            agent: self._getDroneStateVector(drone_name=agent)[10:13]
            for agent in self.all_agents
        }

        colors = [[1, 0, 0], [1, 1, 1], [0, 0, 1], [0,0,0]]  # Red, White, Blue, Black
        path_width = 3
        path_lifetime = 10  # 0 means forever
        update_frequency = 3  # Update path every 10 steps

        for agent in self.all_agents:
            if not self.drone_states[agent]['failed']:
                # Get positions of other drones
                other_positions = [
                    pos for other_agent, pos in all_positions.items()
                    if other_agent != agent
                ]

                other_velocities = [
                    vel for other_agent, vel in all_velocities.items()
                    if other_agent != agent
                ]
                # Update drone's position and navigator
                
                state = self._getDroneStateVector(drone_name=agent)
                drone_pos = state[0:3]

                # print ("drone_pos: ", drone_pos)
                self.navigators[agent].update_drone_position(drone_pos, other_positions, other_velocities)

                # print ("agent: ", agent, " gate: ", self.navigators[agent].current_gate_index)

                # Get next waypoint
                next_waypoint = self.navigators[agent].get_next_waypoint()
                # Update drone's target position
                self.drone_states[agent]['target_position'] = next_waypoint

        # Prepare actions for drones
        action_list = {agent: self.drone_states[agent]['target_position'] for agent in self.all_agents}

        clipped_action = self._preprocessAction(action_list)

        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            
            for agent in self.all_agents:
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[agent], agent)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[agent], agent)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[agent], agent)
                    self._groundEffect(clipped_action[agent], agent)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[agent], agent)
                    self._drag(self.last_clipped_action[agent], agent)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[agent], agent)
                    self._downwash(agent)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[agent], agent)
                    self._groundEffect(clipped_action[agent], agent)
                    self._drag(self.last_clipped_action[agent], agent)
                    self._downwash(agent)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################

        # agent_state, dumb_agent_state = self.observe()
        terminated = {agent: False for agent in self.agents}
        truncated = False
        reward = {agent: 0.0 for agent in self.agents}
        dumb_reward = {agent: 0.0 for agent in self.dumb_agents}
        done = False


        for ind, agent in enumerate(self.all_agents):
            is_agent = agent in self.agents
            current_reward = reward if is_agent else dumb_reward
            navigator = self.navigators[agent]
            gate_index = navigator.current_gate_index
            state = self._getDroneStateVector(drone_name=agent)   
            current_position = np.array(state[0:3])

            
            if agent not in self.position_history:
                self.position_history[agent] = deque(maxlen=self.velocity_window + 1)
                self.velocity_history[agent] = deque(maxlen=self.velocity_window)

            self.position_history[agent].append(current_position)

            # position_list = list(self.position_history[agent])

            if len(self.position_history[agent]) > 1:
                start_pos = self.position_history[agent][-2]
                end_pos = self.position_history[agent][-1]

                if self.timesteps % update_frequency == 0: 
                    p.addUserDebugLine(start_pos, end_pos, colors[ind], path_width, path_lifetime)

                # if self.timesteps % update_frequency == 0:
                #     self.update_path_visualization(agent)


            if agent not in self.position_history:
                self.position_history[agent] = deque(maxlen=self.velocity_window + 1)
                self.velocity_history[agent] = deque(maxlen=self.velocity_window)

            self.position_history[agent].append(current_position)

            if len(self.position_history[agent]) > 1:
                # Calculate velocity
                displacement = np.array(current_position) - np.array(self.position_history[agent][-2])
                norm_disp = np.linalg.norm(displacement)
                velocity = norm_disp / self.time_interval
                # velocity = displacement / self.time_interval
                self.velocity_history[agent].append(velocity)
                mean_velocity = np.mean(self.velocity_history[agent], axis=0)
                self.mean_velocities[agent] = mean_velocity

                # You can add this to your info dict
                self.info[agent]["mean_velocity"] = mean_velocity
                self.info[agent]["instantaneous_velocity"] = velocity

            if self._computeDroneFailed(agent):
                current_reward[agent] = -1.0
                self.info[agent]["successful_flight"] = False
                self.drone_states[agent]['failed'] = True
                if is_agent:
                    terminated[agent] = True
            else:
                if self.n_completion[agent] < navigator.completed_laps:
                    completion_lap_time = self.timesteps - self.previous_lap_time[agent]
                    reward_value = 100 / completion_lap_time
                    current_reward[agent] += reward_value
                    # print (f"drone {agent} completed the track. Reward: {reward_value:.4f}")
                    self.n_completion[agent] = navigator.completed_laps
                    self.previous_lap_time[agent] = self.timesteps
                    self.info[agent]["lap_time"].append(completion_lap_time / self.PYB_FREQ)
                    self.info[agent]["velocity"].append(self.info[agent]["mean_velocity"])

                    # print (f"Lap time: ", self.info[agent]["lap_time"],  "Velocity: ", self.info[agent]["velocity"])
                elif gate_index > self.prev_gate_index[agent]:
                    # print (f"drone {agent} reached the gate {self.prev_gate_index[agent]}")
                    current_reward[agent] += 2.5
                    self.prev_gate_index[agent] = gate_index

                    
                
                reward_val = self._computeReward(drone_name=agent)
                current_reward[agent] += reward_val


        if self.timesteps >= self.MAX_TIMESTEPS:
            truncated = True
  

        
        if all(terminated.values()) or truncated:
            done = True

        # time.sleep(0.01)

        # print (f"drone_pos: {drone_pos[0]:.3f} {drone_pos[1]:.3f} {drone_pos[2]:.3f}")
        # print ("gate_index: ", gate_index, " position: ", self.gate_positions[gate_index])

        reward_val = sum(reward.values())
        dumb_reward_val = sum(dumb_reward.values())

        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)


        return state, reward_val, dumb_reward_val, done, self.info
    

    # def update_path_visualization(self, agent):
    #     positions = list(self.position_history[agent])
    #     if len(positions) < 2:
    #         return

    #     if self.path_item_ids[agent] is not None:
    #         p.removeUserDebugItem(self.path_item_ids[agent])

    #     self.path_item_ids[agent] = p.addUserDebugLine(
    #         positions[0],
    #         positions[-1],
    #         self.path_colors,
    #         lineWidth=self.line_width,
    #         lifeTime=0,
    #         replaceItemUniqueId=self.path_item_ids[agent]
    #     )

    def update_path_visualization(self, agent):
        positions = list(self.position_history[agent])
        if len(positions) < 2:
            return

        # Remove old lines
        for item_id in self.path_item_ids[agent]:
            p.removeUserDebugItem(item_id)
        self.path_item_ids[agent].clear()

        # Add new lines
        for i in range(len(positions) - 1):
            color = self.path_colors[self.color_index[agent]]
            item_id = p.addUserDebugLine(
                positions[i],
                positions[i+1],
                color,
                lineWidth=self.line_width
            )
            self.path_item_ids[agent].append(item_id)
            self.color_index[agent] = (self.color_index[agent] + 1) % len(self.path_colors)

    

def initialize_drones(all_agents, original_init_pos, margin=0.5, altitude=1.0, max_offset=0.15):
    # np.random.seed(int(time.time()))

    drone_positions = {}
    num_drones = len(all_agents)

    # print ("num_drones: ", num_drones)
    # Calculate the angle between each drone on the circle
    angle_step = 2 * np.pi / num_drones
    
    # Calculate the radius of the circle based on the margin between drones
    radius = margin * num_drones / (2 * np.pi)
    
    for i, agent in enumerate(all_agents):
        # Calculate the angle for the current drone
        angle = i * angle_step
        
        # Calculate the x and y coordinates of the drone on the circle
        x = original_init_pos[0] + radius * np.cos(angle)
        y = original_init_pos[1] + radius * np.sin(angle)
        
        # Add a small random offset to the x and y coordinates
        x += np.random.uniform(-max_offset, max_offset)
        y += np.random.uniform(-max_offset, max_offset)
        
        # Set the z coordinate (altitude) of the drone
        z = np.random.uniform(0, altitude)
        
        # Append the drone's position to the list
        drone_positions[agent] = [x, y, z]
    
    return drone_positions