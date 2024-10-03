"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
from train_ppo import PPO
from train_td3 import TD3
from train_ddpg import DDPG
from train_sac import SAC
from common.env_util import make_vec_env
from common.callbacks_updated import EvalCallback, StopTrainingOnRewardThreshold
from common.evaluation import evaluate_policy
from common.monitor import Monitor
from common.vec_env.vec_monitor import VecMonitor
# from envs.MultiGates_SelfPlay_v0 import MultiGates_v0
# from envs.MultiGates_SelfPlay_v1 import MultiGates_v1
from envs.MultiGates_SEIBR import MultiGates_SEIBR
# from gym_pybullet_drones.envs.MultiGatesCont import MultiGatesCont
from utils.utils import sync, str2bool
from utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
MAX_TIMESTEPS = 8000
ENV_NAME = MultiGates_SEIBR

NUM_DRONES = 2
NUM_DUMB_DRONES = 0
TRACK = 1
N_EPISODES = 50

def run_episode(env):
    total_reward = 0
    total_dumb_reward = 0

    env.reset()
    step = 0
    while(True):
        next_state, reward_val, dumb_reward_val, done, info = env.step()
        total_reward += reward_val
        total_dumb_reward += dumb_reward_val
        step += 1
        if done:
            break

    print(f"Completed in {step} steps. Total reward: {total_reward:.4f} Total dumb reward: {total_dumb_reward:.4f}")

    return total_reward, total_dumb_reward



def run():
    
    env = ENV_NAME(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, 
                    gui=DEFAULT_GUI, max_timesteps=MAX_TIMESTEPS, track=TRACK)
    
    total_reward, total_dumb_reward = run_episode(env)
    
    

    

if __name__ == '__main__':
    run()
