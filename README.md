# Multi-Drone Racing Environment

This repository contains a multi-drone racing environment built using the Stable Baselines3 library. It allows for training agents using self-play and various reinforcement learning algorithms.

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
   git clone https://github.com/AkgunOnur/Multi_Drone_Racing.git
2. Navigate to the project directory:
3. Create a new conda environment using the provided environment.yml file:
4. Activate the newly created environment:

## Training

To start training the agents, use the `selfplay.py` file. You can modify various parameters in this file to customize the training process:

- `NUM_DRONES`: Specify the number of drones to use in the environment.
- `NUM_DUMB_DRONES`: Specify the number of opponent drones to use in the environment.
- `ALGO`: Choose the reinforcement learning algorithm to use for training. Available options include PPO, DDPG, SAC, and TD3.
- `TRACK`: Select the racing track to use for training. Set it to 0 for an easier track or 1 for a more challenging track.

Please note that only the PPO algorithm allows for discrete actions. If you want to use DDPG, SAC, or TD3, make sure to set the `discrete_action` parameter to `False`.

Example usage:
```python
# Modify the parameters in selfplay.py
NUM_DRONES = 2
NUM_DUMB_DRONES = 1
ALGO = PPO
TRACK = 1  # Use the more difficult track

# Run the training script
python selfplay.py
```

## Environment Details

The implementation details of the multi-drone racing environment can be found in the envs folder. This folder contains the necessary files and classes for defining the environment dynamics, rewards, and observations.

This project utilizes the Stable Baselines3 library, which provides a set of reliable implementations of reinforcement learning algorithms. However, the code has been modified to support the self-play approach.

For more information about Stable Baselines3, please refer to the official documentation