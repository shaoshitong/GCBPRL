import random

import gym
import numpy as np
import torch

from Environments.FeatureExtract import FE
from Environments.PBS import EnvironmentPBS
from Methods.rl_utils import *
from Methods.SAC import *

path = "./Datas/data2.xlsx"
queue_len = 318
env = EnvironmentPBS(queue_len, path)
choose = env.begin_choose_action
fe = FE(queue_len)

actor_lr = 1e-3 * 2
critic_lr = 1e-2 * 2
alpha_lr = 1e-2 * 2

num_episodes = 200
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 50000
minimal_size = 600
batch_size = 512
target_entropy = -14
action_dim = 14

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
agent = SAC(
    queue_len,
    hidden_dim,
    action_dim,
    action_dim,
    actor_lr,
    critic_lr,
    alpha_lr,
    target_entropy,
    tau,
    gamma,
    device,
)
return_list = train_off_policy_agent(
    env, agent, num_episodes, replay_buffer, minimal_size, batch_size
)
#
# while True:
#     agent_1_action, agent_2_action = choose
#     agent_1_action = np.argwhere(agent_1_action == 1)[:, 0]
#     agent_2_action = np.argwhere(agent_2_action == 1)[:, 0]
#     # print(agent_1_action, agent_2_action)
#     if agent_1_action.size > 0:
#         truth_agent_1_action = np.random.choice(agent_1_action, 1, False).item()
#     else:
#         truth_agent_1_action = -1
#     if agent_2_action.size > 0:
#         truth_agent_2_action = np.random.choice(agent_2_action, 1, False).item()
#     else:
#         truth_agent_2_action = -1
#     # print(truth_agent_1_action, truth_agent_2_action)
#     obs, reward, done, choose = env.step([truth_agent_1_action, truth_agent_2_action])
#     obs, reward, done, choose_2 = fe(obs, reward, done, choose)
#     print(reward, done, choose_2[0].shape)
#     for o in obs:
#         print(o.shape)
#     if done:
#         break
