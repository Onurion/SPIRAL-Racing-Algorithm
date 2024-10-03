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
import time
import numpy as np
import argparse
from envs.MultiGates_SEIBR import MultiGates_SEIBR

DEFAULT_GUI = False
MAX_TIMESTEPS = 8000
ENV_NAME = MultiGates_SEIBR
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

    # print(f"Completed in {step} steps. Total reward: {total_reward:.4f} Total dumb reward: {total_dumb_reward:.4f}")

    return total_reward, total_dumb_reward, info


def run(num_drones, num_dumb_drones, track):

    
    env = ENV_NAME(num_drones=num_drones, num_dumb_drones=num_dumb_drones, 
                    gui=DEFAULT_GUI, max_timesteps=MAX_TIMESTEPS, track=track)
    
    reward_list, dumb_reward_list, info_list = [], [], []

    for i in range(N_EPISODES):
        reward, dumb_reward, info = run_episode(env)
        reward_list.append(reward)
        dumb_reward_list.append(dumb_reward)
        info_list.append(info)

    mean_reward, std_reward = np.mean(reward_list), np.std(reward_list)
    mean_dumb_reward, std_dumb_reward = np.mean(dumb_reward_list), np.std(dumb_reward_list)

    print (f"Mean Reward: {mean_reward:.4f} Std Reward: {std_reward:.4f} Mean Opponent Reward: {mean_dumb_reward:.4f} Std Opponent Reward: {std_dumb_reward:.4f} ")
    # print (info_list)

    result = {}

    for item in info_list:
        for key, value in item.items():
            if 'drone' in key and isinstance(value, dict):
                if key not in result:
                    result[key] = {'lap_time': [], 'successful_flight': []}
                if 'lap_time' in value:
                    result[key]['lap_time'].extend(value['lap_time'])
                if 'successful_flight' in value:
                    result[key]['successful_flight'].append(value['successful_flight'])

    print(result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--num_drones',        default=0,           type=int,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--num_dumb_drones',   default=0,           type=int,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--track',             default=0,           type=int,           help='Folder where to save logs (default: "results")', metavar='')
    args = parser.parse_args()

    run(args.num_drones, args.num_dumb_drones, args.track)
