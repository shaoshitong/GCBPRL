import collections
import random

import numpy as np
import torch
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, choose_action, next_state, done):
        self.buffer.append((self.trun_list_gpu_to_cpu(state), action, reward, self.trun_list_gpu_to_cpu(choose_action),
                            self.trun_list_gpu_to_cpu(next_state), done))

    def trun_list_gpu_to_cpu(self, l):
        new_l = []
        for i in l:
            new_l.append(i.clone().detach().cpu())
        return new_l

    def state_get(self, state):
        new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3 = [], [], [], [], [], []
        for [obs_1, obs_2, obs_3, obs_time_1, obs_time_2, obs_time_3] in state:
            new_obs_1.append(obs_1)
            new_obs_2.append(obs_2)
            new_obs_3.append(obs_3)
            new_obs_time_1.append(obs_time_1)
            new_obs_time_2.append(obs_time_2)
            new_obs_time_3.append(obs_time_3)
        new_obs_1 = torch.stack(new_obs_1)
        new_obs_2 = torch.stack(new_obs_2)
        new_obs_3 = torch.stack(new_obs_3)
        new_obs_time_1 = torch.stack(new_obs_time_1)
        new_obs_time_2 = torch.stack(new_obs_time_2)
        new_obs_time_3 = torch.stack(new_obs_time_3)
        return [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3]

    def sample(self, batch_size):
        begin = random.randint(0,max(0,len(self.buffer)-batch_size-1))
        transitions = []
        for i in range(begin,begin+batch_size):
            transitions.append(self.buffer[i])
        # transitions = self.buffer[begin:begin+batch_size]
        state, action, reward, choose_action, next_state, done = zip(*transitions)
        new_choose_action_1, new_choose_action_2 = [], []
        for choose_action_1, choose_action_2 in choose_action:
            new_choose_action_1.append(choose_action_1)
            new_choose_action_2.append(choose_action_2)
        new_choose_action_1 = torch.stack(new_choose_action_1)
        new_choose_action_2 = torch.stack(new_choose_action_2)
        return (
            self.state_get(state),
            np.array(action),
            np.array(reward),
            [new_choose_action_1, new_choose_action_2],
            self.state_get(next_state),
            np.array(done),
        )

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[: window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


from Environments.PBS import EnvironmentPBS


def train_off_policy_agent(env: EnvironmentPBS, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    from Environments.FeatureExtract import FE
    fe = FE(agent.queue_len)
    num_iter = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _1, _2, choose_action = env.reset(env.path)
                state, _1, _2, choose_action = fe(state, _1, _2, choose_action)
                done = False
                episode_iter = 0
                while not done:
                    action = agent.take_action(state, choose_action)
                    next_state, reward, done, choose_action = env.step(action)
                    next_state, reward, done, choose_action = fe(next_state, reward, done, choose_action)
                    replay_buffer.add(state, action, reward, choose_action, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        if episode_iter%1000==0:
                            print(env.observation,env.observation_require_time,env.current_step)
                        episode_iter+=1
                        b_s, b_a, b_r, b_c, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'choose_action': b_c, 'dones': b_d}
                        c_1_loss, c_2_loss, a_loss = agent.update(transition_dict)
                return_list.append(episode_return)
                print(return_list)
                agent.save(f"model_{num_iter}.pth")
                num_iter += 1
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
