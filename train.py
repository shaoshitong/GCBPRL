import random

import gym
import numpy as np
import torch

from Methods.rl_utils import *
from Methods.SAC import *

actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 500
batch_size = 64
target_entropy = -1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v0"
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = SAC(
    state_dim,
    hidden_dim,
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
