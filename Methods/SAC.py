import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
from torch.autograd import gradcheck
import torch.nn as nn
from .rl_utils import *


class PolicyNet(torch.nn.Module):
    def __init__(self, queue_num, hidden_dim, action1_dim, action2_dim):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 32, (3, 3), (1, 1), (1, 1), bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(32),
                                  nn.AdaptiveAvgPool2d((3, 5)),
                                  nn.Flatten())  # TODO: 480
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(queue_num * 4 + 6 + 480, hidden_dim))

        self.gru = nn.Sequential(nn.GRU(input_size=hidden_dim,hidden_size=hidden_dim,batch_first=True))
        self.layernorm = nn.LayerNorm(hidden_dim)

        self.fc2 = torch.nn.Linear(hidden_dim, action1_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action2_dim)

    def forward(self, x, choose):
        choose[0], choose[1] = choose[0].cuda(), choose[1].cuda()
        [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3] = x
        [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3] = \
            [new_obs_1.cuda(), new_obs_2.cuda(), new_obs_3.cuda(), new_obs_time_1.cuda(), new_obs_time_2.cuda(),
             new_obs_time_3.cuda()]
        if new_obs_1.ndim < 4:
            new_obs_1 = new_obs_1[None, ...]
        if new_obs_time_1.ndim < 3:
            new_obs_time_1 = new_obs_time_1[None, ...]
        if new_obs_2.ndim < 2:
            new_obs_2, new_obs_3, new_obs_time_2, new_obs_time_3 = new_obs_2[None, ...], new_obs_3[None, ...], \
                                                                   new_obs_time_2[None, ...], new_obs_time_3[None, ...]
            choose[0], choose[1] = choose[0][None, ...], choose[1][None, ...]
        feature_map = torch.cat([new_obs_1, new_obs_time_1[..., None]], -1).permute(0, 3, 1, 2)
        y = self.conv(feature_map)
        x = torch.cat([y, self.flatten(new_obs_2), self.flatten(new_obs_3), self.flatten(new_obs_time_2),
                       self.flatten(new_obs_time_3)], 1)
        x = self.fc1(x)
        b,m = x.shape
        if b>1:
            x = x.view(b//64,64,m)
            x = self.layernorm(self.gru(x)[0]).view(b,m)
        else:
            x = self.layernorm(x)
        choose_action_1, choose_action_2 = choose
        mask1 = - (1 - torch.cat([(choose_action_1.sum(1, keepdim=True) <= 0).float(), choose_action_1], 1)) * 1e6
        mask2 = - (1 - torch.cat([(choose_action_2.sum(1, keepdim=True) <= 0).float(), choose_action_2], 1)) * 1e6
        action1 = (self.fc2(x) + mask1).softmax(1)
        action2 = (self.fc4(x) + mask2).softmax(1)
        return action1, action2


class QValueNet(torch.nn.Module):
    def __init__(self, queue_num, hidden_dim, action1_dim, action2_dim):
        super(QValueNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 32, (3, 3), (1, 1), (1, 1), bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm2d(32),
                                  nn.AdaptiveAvgPool2d((3, 5)),
                                  nn.Flatten())  # TODO: 480
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(queue_num * 4 + 6 + 480, hidden_dim))

        self.gru = nn.Sequential(nn.GRU(input_size=hidden_dim,hidden_size=hidden_dim,batch_first=True))
        self.layernorm = nn.LayerNorm(hidden_dim)

        self.fc2 = torch.nn.Linear(hidden_dim, action1_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action2_dim)


    def forward(self, x, choose):
        choose[0], choose[1] = choose[0].cuda(), choose[1].cuda()
        [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3] = x

        feature_map = torch.cat([new_obs_1, new_obs_time_1.unsqueeze(-1)], -1).permute(0, 3, 1, 2)
        y = self.conv(feature_map)
        x = torch.cat([y, self.flatten(new_obs_2), self.flatten(new_obs_3), self.flatten(new_obs_time_2),
                       self.flatten(new_obs_time_3)], 1)
        x = self.fc1(x)
        b,m = x.shape
        if b > 1:
            x = x.view(b // 64, 64, m)
            x = self.layernorm(self.gru(x)[0]).view(b, m)
        else:
            x = self.layernorm(x)
        choose_action_1, choose_action_2 = choose
        mask1 = torch.cat([(choose_action_1.sum(1, keepdim=True) <= 0).float(), choose_action_1], 1).bool()
        mask2 = torch.cat([(choose_action_2.sum(1, keepdim=True) <= 0).float(), choose_action_2], 1).bool()
        action1 = self.fc2(x).float()
        action2 = self.fc4(x).float()
        action1 = torch.where(mask1,action1,torch.Tensor([0.]).to(action1.device))
        action2 = torch.where(mask2,action2,torch.Tensor([0.]).to(action2.device))
        return action1, action2


class SAC:
    """处理离散动作的SAC算法"""

    def __init__(
            self,
            queue_len,
            hidden_dim,
            action1_dim,
            action2_dim,
            actor_lr,
            critic_lr,
            alpha_lr,
            target_entropy,
            tau,
            gamma,
            device,
    ):
        # 策略网络
        self.actor = PolicyNet(queue_len, hidden_dim, action1_dim, action2_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(queue_len, hidden_dim, action1_dim, action2_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(queue_len, hidden_dim, action1_dim, action2_dim).to(device)
        self.target_critic_1 = QValueNet(queue_len, hidden_dim, action1_dim, action2_dim).to(
            device
        )  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(queue_len, hidden_dim, action1_dim, action2_dim).to(
            device
        )  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.queue_len = queue_len

    def take_action(self, state, choose_action):
        probs1, probs2 = self.actor(state, choose_action)
        action_dist1 = torch.distributions.Categorical(probs1)
        action1 = action_dist1.sample()
        action_dist2 = torch.distributions.Categorical(probs2)
        action2 = action_dist2.sample()
        return action1.item() - 1, action2.item() - 1

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, choose_action, dones):
        next_probs1, next_probs2 = self.actor(next_states, choose_action)
        next_probs = torch.cat([next_probs1, next_probs2], 1)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value_1, q1_value_2 = self.target_critic_1(next_states, choose_action)
        q1_value = torch.cat([q1_value_1, q1_value_2], 1)
        q2_value_1, q2_value_2 = self.target_critic_2(next_states, choose_action)
        q2_value = torch.cat([q2_value_1, q2_value_2], 1)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = transition_dict["states"]
        choose_action = transition_dict['choose_action']

        actions1 = (
            torch.tensor(transition_dict["actions"][:, 0]).view(-1, 1).to(self.device).long()
        )  # 动作不再是float类型
        actions2 = torch.tensor(transition_dict["actions"][:, 1]).view(-1, 1).to(self.device).long()
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
                .sum(0, keepdim=True)
                .view(-1, 1)
                .to(self.device)
        )
        next_states = transition_dict["next_states"]
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
                .view(-1, 1)
                .to(self.device)
                .float()
        )
        # 更新两个Q网络
        [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3] = states
        [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3] = \
            [new_obs_1.cuda(), new_obs_2.cuda(), new_obs_3.cuda(), new_obs_time_1.cuda(), new_obs_time_2.cuda(),
             new_obs_time_3.cuda()]
        states = [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3]

        [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3] = next_states
        [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3] = \
            [new_obs_1.cuda(), new_obs_2.cuda(), new_obs_3.cuda(), new_obs_time_1.cuda(), new_obs_time_2.cuda(),
             new_obs_time_3.cuda()]
        next_states = [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3]

        td_target = self.calc_target(rewards, next_states, choose_action, dones)
        q_values = self.critic_1(states, choose_action)
        critic_1_q_values_1 = q_values[0].gather(1, actions1+1)
        critic_1_q_values_2 = q_values[1].gather(1, actions2+1)
        critic_1_q_values = critic_1_q_values_1 + critic_1_q_values_2
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        q_values = self.critic_2(states, choose_action)
        critic_2_q_values_1 = q_values[0].gather(1, actions1+1)
        critic_2_q_values_2 = q_values[1].gather(1, actions2+1)
        critic_2_q_values = critic_2_q_values_1 + critic_2_q_values_2
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs1, probs2 = self.actor(states,choose_action)
        probs = torch.cat([probs1, probs2], 1)
        log_probs = torch.log(probs + 1e-8)
        q1_value = torch.cat(self.critic_1(states,choose_action), 1)
        q2_value = torch.cat(self.critic_2(states,choose_action), 1)
        _,l = q1_value.shape
        # 直接根据概率计算熵
        entropy1 = -torch.sum(probs[:,:l//2] * log_probs[:,:l//2], dim=1, keepdim=True)  #
        entropy2 = -torch.sum(probs[:,l//2:] * log_probs[:,l//2:], dim=1, keepdim=True)  #
        min_qvalue1 = torch.sum(
            probs[:,:l//2] * torch.min(q1_value[:,:l//2], q2_value[:,:l//2]), dim=1, keepdim=True
        )  # 直接根据概率计算期望
        min_qvalue2 = torch.sum(
            probs[:,l//2:] * torch.min(q1_value[:,l//2:], q2_value[:,l//2:]), dim=1, keepdim=True
        )  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy1 - min_qvalue1) + torch.mean(-self.log_alpha.exp() * entropy2 - min_qvalue2)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy1 - self.target_entropy).detach() * self.log_alpha.exp()) + torch.mean((entropy2 - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        return critic_1_loss.cpu().item(), critic_2_loss.cpu().item(), alpha_loss.cpu().item()

    def save(self,path):
        # 策略网络
        dict = {"actor":self.actor.state_dict(),
         "critic_1":self.critic_1.state_dict(),
         "critic_2":self.critic_2.state_dict(),
        "target_critic_1":self.target_critic_1.state_dict(),
        "target_critic_2":self.target_critic_2.state_dict()}
        torch.save(dict,path)

    def load(self,path):
        dict = torch.load(path)
        self.actor.load_state_dict(dict['actor'])
        self.critic_1.load_state_dict(dict['critic_1'])
        self.critic_2.load_state_dict(dict['critic_2'])
        self.target_critic_1.load_state_dict(dict['target_critic_1'])
        self.target_critic_2.load_state_dict(dict['target_critic_2'])