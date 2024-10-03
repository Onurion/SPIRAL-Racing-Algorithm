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
from envs.MultiGates_SelfPlay_v2 import MultiGates_v2
# from gym_pybullet_drones.envs.MultiGatesCont import MultiGatesCont
from utils.utils import sync, str2bool
from utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
MAX_TIMESTEPS = 8000
ENV_NAME = MultiGates_v2

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pos') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
ALGO = PPO
DISCRETE_ACTION = False
NUM_DRONES = 2
NUM_DUMB_DRONES = 2
TRACK = 1
INIT_TYPE = "simple"
MODE_TYPE = "normal"
N_EPISODES = 10


def run():
        
    filename = "SuccessfulModels/25Jul_agent_2_dumb_agent_2_track_1_ppo_pos_continuous_iter_0"

    record_folder = "record_2"
    record = False

    model_to_be_loaded = os.path.join(filename, 'best_model.zip')

    env = ENV_NAME(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, obs=DEFAULT_OBS,  act=DEFAULT_ACT, 
                    gui=DEFAULT_GUI, max_timesteps=MAX_TIMESTEPS, track=TRACK, mode=MODE_TYPE, init_type=INIT_TYPE, record=record, record_folder=record_folder)
    

    if os.path.exists(model_to_be_loaded):
        # Load the saved model
        agent = ALGO.load(model_to_be_loaded, env=env)
        
        print(f"\n\nLoaded previous model from: {model_to_be_loaded} Mode: {MODE_TYPE} Init: {INIT_TYPE}")
        
    else:
        print(f"No such file found! ", model_to_be_loaded)
        agent = ALGO('MlpPolicy',
                    env,
                    tensorboard_log=filename+'/tb/',
                    verbose=1,
                    opponent_model=True,
                    )
        

    mean_reward, std_reward, mean_dumb_reward, std_dumb_reward, info_list = evaluate_policy(model=agent, env=env, n_eval_episodes=N_EPISODES, opponent_policy=agent)
    print (f"Mean Reward: {mean_reward:.4f} Std Reward: {std_reward:.4f} Mean Opponent Reward: {mean_dumb_reward:.4f} Std Opponent Reward: {std_dumb_reward:.4f} ")

        
        
        
    

    

if __name__ == '__main__':
    run()
