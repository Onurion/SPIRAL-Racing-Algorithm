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
from common.callbacks_updated import EvalCallback
from common.evaluation import evaluate_policy
from common.monitor import Monitor
from envs.MultiGates_SelfPlay_v2 import MultiGates_v2
from utils.utils import sync, str2bool
from utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pos') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
ALGO = PPO
DISCRETE_ACTION = True
ENV_NAME = MultiGates_v2
N_ENVS = 4
MAX_TIMESTEPS = 8000
ITER = 1
EVAL_FREQ = int(8000)
TRACK = 1
MODE = "normal"


def get_unique_filename(base_filename):
    counter = 1
    filename = base_filename
    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}"
        counter += 1
    return filename
    

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO):

    reset_num_timesteps, tb_log_name, progress_bar = True, ALGO.__name__, False
    algo_name = ALGO.__name__.lower()

    iteration, log_interval = 0, 100
    policy_kwargs = dict(net_arch=[256, 256])

    kwargs = {"policy_kwargs": policy_kwargs}

    current_date = datetime.now().strftime("%d%b")

    N_drones = [2]
    N_dumb_drones = [2]
    timesteps = [5e7]
    # previous_load_folder = "23Jul_agent_2_dumb_agent_1_track_1_ppo_pos_discrete_iter_"
    previous_load_folder = ""

    for train_index in range(0, len(N_drones)):

        NUM_DRONES = N_drones[train_index]
        NUM_DUMB_DRONES = N_dumb_drones[train_index]
        total_timesteps = timesteps[train_index]

        
        action_type = DEFAULT_ACT.value
        action_space = "discrete" if DISCRETE_ACTION else "continuous"
        # update = "no_update" if NO_MODEL else "update"
        base_filename = f"{current_date}_mode_{MODE}_agent_{NUM_DRONES}_dumb_agent_{NUM_DUMB_DRONES}_track_{TRACK}_{algo_name}_{action_type}_{action_space}_iter_"
        

        for train_it in range(ITER):
            filename = os.path.join(output_folder, algo_name, base_filename + str(train_it))
            filename = get_unique_filename(filename)

            load_folder = os.path.join(output_folder, algo_name, previous_load_folder + str(train_it))
            model_to_be_loaded = os.path.join(load_folder, 'best_model.zip')
            
            if not os.path.exists(filename):
                os.makedirs(filename+'/')

        
            train_env = make_vec_env(ENV_NAME,
                                    env_kwargs=dict(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, obs=DEFAULT_OBS,  act=DEFAULT_ACT, 
                                                    gui=gui, max_timesteps=MAX_TIMESTEPS, track=TRACK, mode=MODE),
                                    n_envs=N_ENVS,
                                    seed=0,
                                    monitor_dir=filename + "/train"
                                    )
            
            # eval_env = make_vec_env(ENV_NAME,
            #                         env_kwargs=dict(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, obs=DEFAULT_OBS,  act=DEFAULT_ACT, gui=gui, max_timesteps=MAX_TIMESTEPS, dumb_no_model=DUMB_NO_MODEL),
            #                         n_envs=N_ENVS,
            #                         seed=0,
            #                         monitor_dir=filename + "/eval"
            #                         )

            eval_env = Monitor(ENV_NAME(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, discrete_action=DISCRETE_ACTION, obs=DEFAULT_OBS, act=DEFAULT_ACT, 
                                        max_timesteps=MAX_TIMESTEPS, track=TRACK, mode=MODE), 
                                        filename=filename + "/eval"
                                        )

            opponent_model = False if NUM_DUMB_DRONES == 0 else True

            # print ("opponent_model: ", opponent_model)

            agent = ALGO('MlpPolicy',
                        train_env,
                        tensorboard_log=filename+'/tb/',
                        verbose=1,
                        opponent_model=True,
                        **kwargs)
            
            if os.path.exists(model_to_be_loaded):
                # Load the saved model

                agent = ALGO.load(model_to_be_loaded, env=train_env)
                agent.tensorboard_log = filename + '/tb/'

                # Set the mlp_extractor of the new model to the loaded mlp_extractor
                # agent.policy.mlp_extractor = loaded_model.policy.mlp_extractor
                # Load the actor and critic networks from the loaded model
                
                # agent.actor.load_state_dict(loaded_model.actor.state_dict())
                # agent.critic.load_state_dict(loaded_model.critic.state_dict())

                # Optionally, you can also load the target networks
                # agent.actor_target.load_state_dict(loaded_model.actor_target.state_dict())
                # agent.critic_target.load_state_dict(loaded_model.critic_target.state_dict())
                
                print(f"Loaded previous model from: {model_to_be_loaded} \n")

            else:
                print(f"No file found! ", model_to_be_loaded, "\n")
                
                
            
            eval_callback = EvalCallback(eval_env,
                                        verbose=1,
                                        best_model_save_path=filename+'/',
                                        log_path=filename+'/',
                                        eval_freq=EVAL_FREQ,
                                        deterministic=True,
                                        render=False,
                                        opponent_model_update=opponent_model)
        
            

            if algo_name == "ppo":
                total_timesteps, callback = agent._setup_learn(total_timesteps, eval_callback, reset_num_timesteps, tb_log_name, progress_bar,)

                callback.on_training_start(locals(), globals())

                while agent.num_timesteps < total_timesteps:
                    continue_training = agent.collect_rollouts(agent.env, callback, agent.rollout_buffer, n_rollout_steps=agent.n_steps)

                    if not continue_training:
                        break

                    iteration += 1
                    agent._update_current_progress_remaining(agent.num_timesteps, total_timesteps)

                    # Display training infos
                    if log_interval is not None and iteration % log_interval == 0:
                        assert agent.ep_info_buffer is not None
                        agent._dump_logs(iteration)

                    agent.train()

                callback.on_training_end()

            elif algo_name == "td3" or algo_name == "ddpg" or algo_name == "sac":

                total_timesteps, callback = agent._setup_learn(total_timesteps,eval_callback,reset_num_timesteps,tb_log_name,progress_bar,)

                callback.on_training_start(locals(), globals())

                while agent.num_timesteps < total_timesteps:
                    rollout = agent.collect_rollouts(
                        agent.env,
                        train_freq=agent.train_freq,
                        action_noise=agent.action_noise,
                        callback=callback,
                        learning_starts=agent.learning_starts,
                        replay_buffer=agent.replay_buffer,
                        log_interval=log_interval,
                    )

                    if not rollout.continue_training:
                        break

                    if agent.num_timesteps > 0 and agent.num_timesteps > agent.learning_starts:
                        # If no `gradient_steps` is specified,
                        # do as many gradients steps as steps performed during the rollout
                        gradient_steps = agent.gradient_steps if agent.gradient_steps >= 0 else rollout.episode_timesteps
                        # Special case when the user passes `gradient_steps=0`
                        if gradient_steps > 0:
                            agent.train(batch_size=agent.batch_size, gradient_steps=gradient_steps)

                callback.on_training_end()


        
        previous_load_folder = base_filename


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
