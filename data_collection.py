import sys
sys.path.append('/home/onur/Downloads/MultiDrone/gym_drones')
sys.path.append('/home/onur/Downloads/MultiDrone/sb3_selfplay')

import os
import pickle
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
from common.buffers_updated import ReplayBuffer
from td3.policies import TD3Policy
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from datetime import datetime

from envs.MultiGates_SelfPlay_v1 import MultiGates_v1
from envs.MultiGates_SelfPlay_v2 import MultiGates_v2
from envs.MultiGates_SelfPlay_v3 import MultiGates_v3
from train_ppo import PPO
from train_td3 import TD3
from train_ddpg import DDPG
from train_sac import SAC
from common.evaluation import evaluate_policy, evaluate_policy_noselfplay
from common.env_util import make_vec_env, make_vec_env_nomonitor
# from common.utils import update_learning_rate
from common.utils import get_parameters_by_name, polyak_update


DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pos') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
ENV_NAME = MultiGates_v3
MAX_TIMESTEPS = 8000
NUM_DRONES = 2
NUM_DUMB_DRONES = 0
TRACK = 2
N_DEMOS = 2000 
N_EPOCHS = 500

def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def update_learning_rate(optimizers, learning_rate):
    """
    Update the learning rate for all optimizers
    :param optimizers: list of optimizers
    :param learning_rate: new learning rate
    """
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

def train():
    # Create the environment
    act = None 
    output_folder = "results/imitation_learning"
    action_type = DEFAULT_ACT.value
    algo_list = ["td3", "ddpg"]
    method_list = [0, 1]
    N_training = 4
    device = 'cpu'
    current_date = datetime.now().strftime("%d%b")

    data_collect_env = ENV_NAME(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES, obs=DEFAULT_OBS,  act=DEFAULT_ACT, gui=False, max_timesteps=MAX_TIMESTEPS, 
                                difficulty_level=0, track=TRACK, mode="data_collection")
    
    eval_env = ENV_NAME(num_drones=NUM_DRONES, num_dumb_drones=NUM_DUMB_DRONES,  obs=DEFAULT_OBS,  act=DEFAULT_ACT, gui=False, max_timesteps=MAX_TIMESTEPS, 
                                    difficulty_level=0, track=TRACK, mode="normal")
    
    # Create a replay buffer to store experience
    buffer_size = 1_000_000  # Adjust as needed
    replay_buffer = ReplayBuffer(
        buffer_size,
        eval_env.observation_space,
        eval_env.action_space,
        device=device,
        n_envs=1,
    )

    # Collect expert demonstrations
    print ("Data collection started!")
    for i in range(N_DEMOS):
        obs, dumb_obs,  _ = data_collect_env.reset()
        done = False
        print (f"Episode: {i}/{N_DEMOS}")
        while not done:
            # Step the environment
            expert_action, next_obs, reward, terminated, truncated, _ = data_collect_env.step(act)

            if DEFAULT_ACT == "rpm":
                action_array = np.array(list(expert_action.values())) / 1e5
            else:
                action_array = np.array(list(expert_action.values()))

            # print ("action_array: ", action_array)

            done = terminated or truncated
            
            # Add to replay buffer
            replay_buffer.add(obs, next_obs, action_array, reward, done, truncated)
            
            obs = next_obs

    # Save the replay buffer to a file
    # with open(f"{current_date}_replay_buffer.pkl", 'wb') as file:
    #     pickle.dump(replay_buffer, file)

    print ("Data collection is over! \n")

    for iteration in range(N_training):
        algo_name = algo_list[iteration % len(algo_list)]
        method = method_list[iteration % len(method_list)]
        
        base_filename = f"{current_date}_imitation_agent_{NUM_DRONES}_track_{TRACK}_{algo_name}_{action_type}"

        filename = os.path.join(output_folder, algo_name, base_filename)

        if not os.path.exists(filename):
            os.makedirs(filename+'/')
        
        # Initialize the TD3 policy
        policy = TD3Policy(
            observation_space=eval_env.observation_space,
            action_space=eval_env.action_space,
            lr_schedule=lambda _: 3e-4,  # You can adjust the learning rate
            net_arch=[400, 300],  # You can adjust the architecture
        )
        
        
        # Imitation Learning
        batch_size = 512  # Adjust as needed
        epoch_freq = 1
        best_reward = -1000.
        n_updates = 0

        if algo_name == "ddpg":
            policy_delay=1
            target_noise_clip=0.0
            target_policy_noise=0.1
        elif algo_name == "td3":
            policy_delay = 2
            target_policy_noise = 0.2
            target_noise_clip = 0.5

        gradient_steps = 1
        gamma = 0.99
        imitation_weight = 0.5
        tau = 0.005

        # Usage in your training loop:
        initial_lr = 1e-3
        final_lr = 1e-4
        lr_schedule = linear_schedule(initial_lr, final_lr)

        if method == 0:
            for epoch in range(N_EPOCHS):

                # Switch to train mode (this affects batch norm / dropout)
                policy.set_training_mode(True)

                _current_progress_remaining = 1.0 - float(epoch) / float(N_EPOCHS)
            
                # Calculate current learning rate
                current_lr = lr_schedule(_current_progress_remaining)
                
                # Update learning rate according to lr schedule
                update_learning_rate([policy.actor.optimizer, policy.critic.optimizer], current_lr)

                actor_batch_norm_stats = get_parameters_by_name(policy.actor, ["running_"])
                critic_batch_norm_stats = get_parameters_by_name(policy.critic, ["running_"])
                actor_batch_norm_stats_target = get_parameters_by_name(policy.actor_target, ["running_"])
                critic_batch_norm_stats_target = get_parameters_by_name(policy.critic_target, ["running_"])

                actor_losses, critic_losses, imitation_losses = [], [], []
                for _ in range(gradient_steps):
                    n_updates += 1
                    # Sample replay buffer
                    replay_data = replay_buffer.sample(batch_size)

                    with th.no_grad():
                        # Select action according to policy and add clipped noise
                        noise = replay_data.actions.clone().data.normal_(0, target_policy_noise)
                        noise = noise.clamp(-target_noise_clip, target_noise_clip)
                        next_actions = (policy.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                        # Compute the next Q-values: min over all critics targets
                        next_q_values = th.cat(policy.critic_target(replay_data.next_observations, next_actions), dim=1)
                        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                        target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

                    # Get current Q-values estimates for each critic network
                    current_q_values = policy.critic(replay_data.observations, replay_data.actions)

                    # Compute critic loss
                    critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                    critic_losses.append(critic_loss.item())

                    # Optimize the critics
                    policy.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    policy.critic.optimizer.step()

                    # Delayed policy updates
                    if n_updates % policy_delay == 0:
                        # Compute actor loss (TD3 loss)
                        td3_actor_loss = -policy.critic.q1_forward(replay_data.observations, policy.actor(replay_data.observations)).mean()
                        
                        # Compute imitation loss
                        predicted_actions = policy.actor(replay_data.observations)
                        imitation_loss = F.mse_loss(predicted_actions, replay_data.actions)
                        imitation_losses.append(imitation_loss.item())
                        
                        # Combine TD3 and imitation losses
                        actor_loss = (1 - imitation_weight) * td3_actor_loss + imitation_weight * imitation_loss
                        actor_losses.append(actor_loss.item())

                        # Optimize the actor
                        policy.actor.optimizer.zero_grad()
                        actor_loss.backward()
                        policy.actor.optimizer.step()

                        polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)
                        polyak_update(policy.actor.parameters(), policy.actor_target.parameters(), tau)
                        # Copy running stats, see GH issue #996
                        polyak_update(critic_batch_norm_stats, critic_batch_norm_stats_target, 1.0)
                        polyak_update(actor_batch_norm_stats, actor_batch_norm_stats_target, 1.0)


                # Evaluate the policy
                if (epoch + 1) % epoch_freq == 0:
                    mean_reward, std_reward = evaluate_policy(policy, eval_env, n_eval_episodes=10)
                    print(f"Algo: {algo_name}_{method} Epoch: {epoch + 1}/{N_EPOCHS}, Average Reward: {mean_reward:.4f} Std Reward: {std_reward:.4f} Best model reward so far: {best_reward:.4f}")
                    if mean_reward > best_reward:
                        # Save the imitation-learned policy
                        th.save(policy.state_dict(), filename + "/policy_method_0.pth")
                        best_reward = mean_reward


        elif method == 1:
            for epoch in range(N_EPOCHS):
                for _ in range(replay_buffer.pos // batch_size):
                    # Sample a batch from the replay buffer
                    data = replay_buffer.sample(batch_size)
                    
                    # Compute actor loss (imitation loss)
                    predicted_actions = policy.actor(data.observations)
                    actor_loss = F.mse_loss(predicted_actions, data.actions)
                    
                    # Optimize the actor
                    policy.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    policy.actor.optimizer.step()
                
                # Evaluate the policy
                if (epoch + 1) % epoch_freq == 0:
                    mean_reward, std_reward = evaluate_policy(policy, eval_env, n_eval_episodes=10)
                    print(f"Algo: {algo_name}_{method} Epoch: {epoch + 1}/{N_EPOCHS}, Average Reward: {mean_reward:.4f} Std Reward: {std_reward:.4f} Best model reward so far: {best_reward:.4f}")
                    if mean_reward > best_reward:
                        # Save the imitation-learned policy
                        th.save(policy.state_dict(), filename + "/policy_method_1.pth")
                        best_reward = mean_reward



def eval():

    eval_env = ENV_NAME(num_drones=1, num_dumb_drones=0, obs=DEFAULT_OBS,  act=DEFAULT_ACT, gui=True, max_timesteps=MAX_TIMESTEPS, 
                        difficulty_level=0, track=2, mode="normal")

    policy = TD3Policy(
        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        lr_schedule=lambda _: 3e-4,  # You can adjust the learning rate
        net_arch=[400, 300],  # You can adjust the architecture
    )


    folder = "results/imitation_learning/td3/20Jul_imitation_agent_1_track_2_td3_pos"
    # Load the saved state dictionary
    policy.load_state_dict(th.load(folder + "/policy_method_0.pth"))

    # Set the policy to evaluation mode
    policy.eval()

    mean_reward, std_reward = evaluate_policy(policy, eval_env, n_eval_episodes=10)
    print(f"Average Reward: {mean_reward:.4f} Std Reward: {std_reward:.4f}")


if __name__ == '__main__':
    # train()
    eval()