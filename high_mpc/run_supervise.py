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
from high_mpc.mpc.mpc import MPC
# nn training
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch

import argparse


def data_collection():
    state_dim = 9
    action_dim = 4
    obs_buf = []
    act_buf = []
    save_dir = "high_mpc/Dataset/"

    ##setting for environment
    plan_T = 2
    plan_dt = 0.1

    so_path = "high_mpc/mpc/saved/mpc_v1.so" # saved mpc model (casadi code generation)
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(mpc, plan_T, plan_dt)
    num_data_samples = 40000
    data_path = save_dir + "/obs_{0}".format(num_data_samples)
    time_step = 0
    while time_step <= num_data_samples:
        obs = env.reset()
        t = 0

        while t < env.sim_T:
            t += env.sim_dt
            obs_buf.append(obs)
            obs, quad_act, terminated, info = env.step_collect_data()
            act_buf.append(quad_act)

            time_step += 1
            if time_step % 1000 == 0:
                print("+1000")

            if terminated:
                break
            if t >= env.sim_T:
                break

    obs_npy = np.array(obs_buf)
    act_npy = np.array(act_buf)
    # save data collection
    np.savez(data_path, obs=obs_npy, act=act_npy)


class Dataset(Dataset):
    def __init__(self, data_file):
        np_file = np.load(data_file)
        self.obs_array = np_file['obs'].astype('float32')
        self.act_array = np_file['act'].astype('float32')

    def __len__(self):
        return len(self.obs_array)

    def __getitem__(self, idx):
        obs = self.obs_array[idx, :]
        label = self.act_array[idx, :]

        return obs, label


class Network(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
        nn.Linear(num_states, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_actions)
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output



def train(train_file, val_file, save_weight_path):
    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 450

    training_data = Dataset(data_file=train_file)
    validate_data = Dataset(data_file=val_file)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)

    model = Network(num_states=18, num_actions=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    for ep in range(num_epochs):
        model.train()
        sum_loss = 0
        best_loss = 100
        for i, (obs, label) in enumerate(train_dataloader):
            pred = model(obs)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sum_loss  += loss.item()
        loss_batch = sum_loss / len(train_dataloader)
        print("Ep: {}, Val loss: {:.3f}".format(ep, loss_batch))
        if ep % 20 == 0:
            torch.save(model.state_dict(), save_weight_path + "ep_{}.pth".format(str(ep)))
        # model.eval()
        # loss_sum = 0
        # best_loss = 100
        # num_batches = len(val_dataloader)
        # with torch.no_grad():
        #     for obs, label in val_dataloader:
        #         pred = model(obs)
        #         val_loss = loss_fn(pred, label)
        #         loss_sum += val_loss.item()
        # loss_batch = loss_sum / num_batches
        # print("Ep: {}, Val loss: {:.2f}".format(ep, loss_batch))
        # if loss_batch < best_loss:
        #     best_loss = loss_batch
        #     torch.save(model.state_dict(), save_weight_path)

def run_exp(env, nn_model):
    print("run")
    obs = env.reset()
    nn_model.eval()
    t = 0
    t0 = time.time()
    while t < env.sim_T:
        t += env.sim_dt
        #_, _, _, info = env.step()
        pred_act = nn_model(torch.from_numpy(np.array(obs).astype("float32")))
        pred_act = np.expand_dims(pred_act.detach().numpy(), axis=1)
        obs, quad_act, terminated, info = env.step_nn(u=0, pred_act=pred_act)


        t_now = time.time()
        #print(t_now - t0)
	    #
        t0 = time.time()
        #
        update = False
        if t >= env.sim_T:
            update = True
        yield [info, t, update]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-collect', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--run', action='store_true')

    args = parser.parse_args()


    if args.data_collect:
        data_collection()
    elif args.training:
        train_file = "high_mpc/Dataset/obs2_40000.npz"
        val_file = "high_mpc/Dataset/data_4000.npz"
        save_weight_path = "high_mpc/weights/obs2_40k/"
        train(train_file, val_file, save_weight_path)
    elif args.run:
        plan_T = 2.0   # Prediction horizon for MPC and local planner
        plan_dt = 0.1 # Sampling time step for MPC and local planner
        so_path = "high_mpc/mpc/saved/mpc_v1.so" # saved mpc model (casadi code generation)
        save_weight_path = "high_mpc/weights/obs2_40k/" + "ep_400.pth"

        mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
        env = DynamicGap(mpc, plan_T, plan_dt)
        nn_model = Network(num_states=18, num_actions=4)
        nn_model.load_state_dict(torch.load(save_weight_path))

        #run_exp(env, nn_model)

        sim_visual = SimVisual(env)
        run_frame = partial(run_exp, env, nn_model)
        ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
            init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)
        if True:
            writer = animation.writers["ffmpeg"]
            writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            ani.save("high_mpc/mpc/saved/output2.mp4", writer=writer)

        plt.tight_layout()
        plt.show()




#
