import os
import time
import numpy as np
import torch
from functools import partial
#
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# A Gym style environment
from high_mpc.simulation.dynamic_gap import DynamicGap
from high_mpc.simulation.animation import SimVisual
from high_mpc.common import logger
from high_mpc.common import util as U
from high_mpc.policy.ppo import PPO
from high_mpc.mpc.mpc import MPC


def train():
    state_dim = 9
    action_dim = 4
    action_std = 0.6
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = int(2.5e5)

    ################ PPO hyperparameters ################

    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    save_dir = "high_mpc/weights/"
    run_num = "gate_1/"

    ##setting for environment
    plan_T = 2
    plan_dt = 0.04

    if not os.path.exists(os.path.join(save_dir, str(run_num))):
        os.mkdir(os.path.join(save_dir, str(run_num)))
    checkpoint_path = save_dir + run_num + "ppo_gate.pth"


    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    so_path = "high_mpc/mpc/saved/mpc_v1.so" # saved mpc model (casadi code generation)
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(mpc, plan_T, plan_dt)

    update_timestep = int(env.sim_T /env.sim_dt) * 4      # update policy every n timesteps
    time_step = 0
    i_episode = 0
    sum_reward = 0
    print_episode = 0
    while time_step <= max_training_timesteps:
        obs = env.reset()
        current_ep_reward = 0
        t = 0
        while t < env.sim_T:
            t += env.sim_dt
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=1)
            obs, reward, terminated, info = env.step_reinforce(action)

             # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(terminated)


            time_step += 1

            current_ep_reward += reward
            # update PPO agent
            if time_step % update_timestep == 0:
                #print("update")
                ppo_agent.update()

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if terminated:
                break
            if t >= env.sim_T:
                break
        i_episode += 1
        sum_reward += current_ep_reward
        print_episode += 1

        if time_step % int(env.sim_T /env.sim_dt) *10 == 0:
            print("Episode: {}, Reward: {}".format(i_episode, round(sum_reward / print_episode, 2)))
            sum_reward = 0
            print_episode = 0








if __name__ == "__main__":
    train()


#
