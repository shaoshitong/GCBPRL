import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random, time

import logging
import random
import gym
from gym import spaces
import copy
import matplotlib.pyplot as plt
import matplotlib as  mpl
from matplotlib  import pyplot as plt
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
logger = logging.getLogger(__name__)


class Car:
    def __init__(self):
        self.type = 0
        self.power = 0
        self.drive = 0
        self.number = 0

    def set(self, type, power, drive, number):
        self.type = type
        self.power = power
        self.drive = drive
        self.number = number


class Queue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = []

    def put(self, item):
        if len(self.queue) < self.maxsize:
            self.queue.append(item)
            return True
        return False

    def top(self):
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            raise KeyError

    def get(self):
        if len(self.queue) > 0:
            top = copy.deepcopy(self.top())
            del self.queue[0]
            return top
        else:
            raise KeyError

    def empty(self):
        return len(self.queue) == 0


class EnvironmentPBS(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, queue_number, path):
        super(EnvironmentPBS, self).__init__()
        self.queue_number = queue_number
        self.path = path
        self.cin_road_number = 6
        self.cout_road_number = 1
        self.reward_range = [1, 2, 1, lambda x, c=queue_number: 100 - 0.01 * max(0, x - 9 * c - 72)]
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(  # TODO: in (13) and out (13)
            low=np.array([0, 0]), high=np.array([13, 13]), dtype=np.int32)
        self.observation_space = spaces.Box(  # TODO: (7 , 10) [PBS的状态]
            low=0, high=1, shape=(7, 10), dtype=np.int32)  # TODO: 假设第七行是返回道
        self.observation_space_2 = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.int32)
        self.observation_space_3 = spaces.Box(low=np.array([0, 0]), high=np.array([100, 100]), dtype=np.int32)

        self.agent_1_require_time_1 = [18, 12, 6, 0, 12, 18]
        self.agent_1_require_time_2 = [24, 18, 12, 6, 12, 18]
        self.agent_2_require_time_1 = [18, 12, 6, 0, 12, 18]
        self.agent_2_require_time_2 = [24, 18, 12, 6, 12, 18]
        self.reset_number = 0
        self.reward_result = []
        self.begin_obs, self.begin_reward, self.begin_done, self.begin_choose_action = self.reset(path)

        print("successfully init environment!")

    def _next_observation(self):  # TODO: get observation
        return copy.deepcopy(self.observation), copy.deepcopy(self.agent_observation), self.tuzhuang_observation

    def read_xlsx_file(self, path):  # TODO: read CSV file
        import pandas as pd
        dataframe = pd.read_excel(path, engine='openpyxl')
        dataframe = dataframe.replace("A", 0)
        dataframe = dataframe.replace("B", 1)
        dataframe = dataframe.replace("燃油", 0)
        dataframe = dataframe.replace("混动", 1)
        dataframe = dataframe.replace("两驱", 0)
        dataframe = dataframe.replace("四驱", 1)
        dataframe = dataframe.astype('int')
        return dataframe

    def push_element_to_queue(self, dataframe):  # TODO: 把dataframe元素放入队列
        dataframe = dataframe.sort_values(by=['进车顺序'], na_position='first')
        for index, row in dataframe.iterrows():
            c, d, q, n = row['车型'], row['动力'], row['驱动'], row['进车顺序']
            car = Car()
            car.set(c, d, q, n)
            self.tuzhuang_observation[0].put(car)

    def reset(self, path: str):  # TODO: 环境重置
        """
        :return: np.array
        """
        self.score = 100
        self.observation = np.zeros((7, 10), dtype=np.int32)
        self.agent_observation = np.zeros((2,), dtype=np.int32)  # TODO: (2) [送车和接车的状态]
        self.tuzhuang_observation = [Queue(maxsize=self.queue_number + 1), Queue(maxsize=self.queue_number + 1)]
        dataframe = self.read_xlsx_file(path)
        self.push_element_to_queue(dataframe)
        self.dictionary = self._queue_to_list(self.tuzhuang_observation[0])

        self.observation_require_time = np.zeros((7, 10), dtype=np.int32)
        self.agent_observation_require_time = np.zeros((2,), dtype=np.int32)  # TODO: (2) [送车和接车的状态]
        self.tuzhuang_observation_require_time = np.zeros((2,), dtype=np.int32)  # TODO: (3) [涂装口和总装口]
        self.current_step = 0
        self.output_array = []
        self.turn_dictionary = self._turn_dictionary()

        self.last_constraint_1 = 40
        self.last_constraint_2 = 30
        self.last_constraint_3 = 20
        self.last_constraint_4 = 10
        self.return_road_used = 0
        self.time_cnt = np.zeros((6,), dtype=np.int32)
        self.reset_number += 1

        choose_action = self._get_choose_action()
        self.current_step += 1
        done = self._done()
        reward = self._reward_compute(done=done)
        obs = self._next_observation()
        obs_1, obs_2 = self._key_to_array(obs[0]), obs[1]
        obs_3 = obs[2]
        obs_3_1 = self._queue_to_array(obs_3[0])
        obs_3_2 = self._queue_to_array(obs_3[1])
        obs_time_1, obs_time_2, obs_time_3 = self.observation_require_time, self.agent_observation_require_time, self.tuzhuang_observation_require_time

        obs = [obs_1, obs_2, obs_3_1, obs_3_2, obs_time_1, obs_time_2, obs_time_3]  # TODO: 缓冲区的状态矩阵, 智能体的状态矩阵,
        # TODO: 出装的状态矩阵，总装的状态矩阵,
        # TODO: (缓冲区的状态矩阵剩余时间, 智能体的状态矩阵剩余时间,出装和总装剩余时间)

        return obs, reward, done, choose_action

    def _turn_dictionary(self):
        import pandas as pd
        dataframe = pd.read_excel("./Datas/data3.xlsx", engine='openpyxl')
        dataframe = dataframe.astype('str')
        dataframe = dataframe.replace("停车位", "", regex=True)
        result = {}
        for index, row in dataframe.iterrows():
            c, d = row['区域'], row['代码']
            if "进车道" in c:
                a, b = c.split("进车道")
                a, b = int(a) - 1, int(b)
                b = 10 - b
                d = int(d)
                result[str(10 + b * 100 + a)] = d
            elif "返回道" in c:
                a = 6
                b = 10 - int(c.replace("返回道", ""))
                d = int(d)
                result[str(10 + b * 100 + a)] = d
            else:
                pass
        return result

    def _every_time_step(self):  # TODO: 环境的单步更新
        now_observation, now_agent_observation, now_tuzhuang_observation = self._next_observation()
        now_observation_require_time = copy.deepcopy(self.observation_require_time)
        now_agent_observation_require_time = copy.deepcopy(self.agent_observation_require_time)
        now_tuzhuang_observation_require_time = copy.deepcopy(self.tuzhuang_observation_require_time)

        # TODO: 处理智能体的进程
        # TODO: 更新出装和总装

        now_agent_observation_require_time = np.clip(now_agent_observation_require_time - 1, 0, 1e6)
        now_tuzhuang_observation_require_time = np.clip(now_tuzhuang_observation_require_time - 1, 0, 1e6)
        agent_1_require_time = (now_agent_observation_require_time[0] > 0).sum().item()
        agent_2_require_time = (now_agent_observation_require_time[1] > 0).sum().item()
        if agent_1_require_time == 0:
            if 0 < now_agent_observation[0] <= 6 and (now_observation[now_agent_observation[0] - 1, 0].item() == 0):
                car = now_tuzhuang_observation[0].get().number
                now_observation[now_agent_observation[0] - 1, 0] = car
                now_observation_require_time[now_agent_observation[0] - 1, 0] = self.agent_1_require_time_1[
                    now_agent_observation[0] - 1]
                now_tuzhuang_observation_require_time[0] = 0
                now_agent_observation_require_time[0] = self.agent_1_require_time_1[now_agent_observation[0] - 1]

            elif 6 < now_agent_observation[0] <= 12 and (now_observation[6, 0].item() != 0) \
                    and (now_observation_require_time[6, 0].item() <= 1) and (
                    now_observation[now_agent_observation[0] - 7, 0].item() == 0):
                car = now_observation[6, 0]
                now_observation[now_agent_observation[0] - 7, 0] = car
                now_observation[6, 0] = 0
                now_observation_require_time[6, 0] = 0
                now_observation_require_time[now_agent_observation[0] - 7, 0] = self.agent_1_require_time_2[
                    now_agent_observation[0] - 7]
                now_agent_observation_require_time[0] = self.agent_1_require_time_2[now_agent_observation[0] - 7]

            elif now_agent_observation[0] == 0:
                pass
            else:
                raise NotImplementedError

        if agent_2_require_time == 0:
            if 0 < now_agent_observation[1] <= 6 and now_observation_require_time[
                now_agent_observation[1] - 1, 9].item() <= 1 \
                    and now_observation[now_agent_observation[1] - 1, 9].item() > 0:
                car_number = now_observation[now_agent_observation[1] - 1, 9]
                type, power, drive = self.dictionary[car_number]
                car = Car()
                car.set(type, power, drive, car_number)
                now_tuzhuang_observation[1].put(car)
                now_tuzhuang_observation_require_time[1] = 9
                now_observation_require_time[now_agent_observation[1] - 1, 9] = 0
                now_observation[now_agent_observation[1] - 1, 9] = 0
                now_agent_observation_require_time[1] = self.agent_2_require_time_1[now_agent_observation[1] - 1]

            elif 6 < now_agent_observation[1] <= 12 and now_observation_require_time[
                now_agent_observation[1] - 7, 9].item() <= 1 \
                    and now_observation[now_agent_observation[1] - 7, 9].item() > 0:
                car = now_observation[now_agent_observation[1] - 7, 9]
                self.return_road_used += 1
                now_observation[now_agent_observation[1] - 7, 9] = 0
                now_observation[6, 9] = car
                now_observation_require_time[now_agent_observation[1] - 7, 9] = 0
                now_observation_require_time[6, 9] = 9
                now_agent_observation_require_time[1] = self.agent_2_require_time_2[now_agent_observation[1] - 7]
            elif now_agent_observation[1] == 0:
                pass
            else:
                raise NotImplementedError

        for i in range(9, 0, -1):
            # TODO: 进车道
            second_observation = now_observation[:6][:, i]
            mask = (second_observation == 0)
            first_observation = now_observation[:6][:, i - 1]
            before_mask = (now_observation_require_time[:6][:, i - 1] - 1 <= 0) & (first_observation > 0)
            mask = mask & before_mask
            if mask.sum().item() > 0:
                now_observation[:6, i][mask] = now_observation[:6, i - 1][mask]
                now_observation[:6, i - 1][mask] = 0
                if i == 9:
                    self.time_cnt[mask] = self.current_step
            if (~mask).sum().item() > 0:
                temp = np.where(0 < now_observation_require_time[:6, i] - 1,
                                now_observation_require_time[:6, i] - 1,
                                np.zeros_like(now_observation_require_time[:6, i]))
                now_observation_require_time[:6, i] = np.where(mask, now_observation_require_time[:6, i], temp)
            now_observation_require_time[:6, i][mask] = 9
            now_observation_require_time[:6, i - 1][mask] = 0
            # TODO: 返回道
            j = 10 - i
            second_observation = now_observation[6, j - 1]
            mask = (second_observation == 0)
            first_observation = now_observation[6, j]
            before_mask = (now_observation_require_time[6, j] - 1 <= 0) & (first_observation > 0)
            mask = mask & before_mask
            if mask:
                now_observation[6, j - 1] = now_observation[6, j]
                now_observation[6, j] = 0
            if not mask:
                if 0 < now_observation_require_time[6, j - 1] - 1:
                    temp = now_observation_require_time[6, j - 1] - 1
                else:
                    temp = np.zeros_like(now_observation_require_time[6, j - 1])
                now_observation_require_time[6, j - 1] = temp
            if mask:
                now_observation_require_time[6, j - 1] = 9
                now_observation_require_time[6, j] = 0
        # TODO: 解决末端计数
        f_mask = (now_observation[:6, 0] > 0)  # TODO: in
        if f_mask.sum().item():
            now_observation_require_time[:6, 0][f_mask] = \
                np.where(now_observation_require_time[:6, 0][f_mask] > 0,
                         now_observation_require_time[:6, 0][f_mask] - 1,
                         now_observation_require_time[:6, 0][f_mask])

        f_mask = (now_observation[6, 9] > 0)  # TODO: out
        if f_mask:
            now_observation_require_time[6, 9] = \
                np.where(now_observation_require_time[6, 9] > 0,
                         now_observation_require_time[6, 9] - 1,
                         now_observation_require_time[6, 9])

        self.observation, self.agent_observation, self.tuzhuang_observation = now_observation, now_agent_observation, now_tuzhuang_observation
        self.observation_require_time, self.agent_observation_require_time, self.tuzhuang_observation_require_time = \
            now_observation_require_time, now_agent_observation_require_time, now_tuzhuang_observation_require_time

    def _done(self):  # TODO: 判断游戏是否结束
        empty = self.tuzhuang_observation[0].empty()
        done = self.observation.sum().item()
        return (empty and (done == 0)) or (self.current_step > 18000)

    def _queue_to_list(self, queue):  # TODO: 队列转换至list
        result = [i for i in range(self.queue_number + 1)]
        for car in queue.queue:
            result[car.number] = [car.type, car.power, car.drive]
        return result

    def _key_to_array(self, key_array):  # TODO: 键值转换为list
        result = np.zeros((7, 10, 4), dtype=np.int32)
        for i in range(7):
            for j in range(10):
                key = key_array[i, j].item()
                if key:
                    type, power, drive = self.dictionary[key]
                    result[i, j, :] = np.array([key, type, power, drive], dtype=np.int32)
                else:
                    result[i, j, :] = np.array([-1, -1, -1, -1], dtype=np.int32)
        return result

    def _queue_to_array(self, queue):  # TODO: 队列转换为 Array
        r = self._queue_to_list(queue)
        l = len(queue.queue)
        result = np.zeros((l, 4), dtype=np.int32)
        point = 0
        for i, m in enumerate(r[1:]):
            if isinstance(m, list):
                type, power, drive = m
                result[point] = np.array([i, type, power, drive], dtype=np.int32)
                point += 1
        return result

    def _every_car_where(self):
        result = np.zeros(self.queue_number)
        for car in self.tuzhuang_observation[0].queue:
            number = car.number
            result[number - 1] = 0
        for car in self.tuzhuang_observation[1].queue:
            number = car.number
            result[number - 1] = 3
        for i in range(7):
            for j in range(10):
                if self.observation[i, j] != 0:
                    if self.observation_require_time[i,j]!=0:
                        if i<6:
                            result[self.observation[i, j].item() - 1] = self.turn_dictionary[str(10 + (j - 1) * 100 + i)]
                        else:
                            result[self.observation[i, j].item() - 1] = self.turn_dictionary[str(10 + (j + 1) * 100 + i)]
                    else:
                        result[self.observation[i, j].item() - 1] = self.turn_dictionary[str(10 + j * 100 + i)]

        if self.agent_observation_require_time[0] > 0:
            index = self.agent_observation[0] - 7 if self.agent_observation[0] >= 7 else self.agent_observation[0] - 1
            result[self.observation[index, 0].item() - 1] = 1
        if self.agent_observation_require_time[1] > 0:
            if self.agent_observation[1] >= 7:
                result[self.observation[6, 9].item() - 1] = 2
            else:
                result[self.tuzhuang_observation[1].queue[-1].number - 1] = 2
        return result

    def _reward_compute(self, done):  # TODO: 计算当前回报 (相较于上一次回报获取，增加或者减少了多少)
        output_queue = copy.deepcopy(self.tuzhuang_observation[1])
        num = 0
        constraint_1 = 100
        constraint_2 = 100
        constraint_3 = 100
        constraint_1_tag = 0
        constraint_3 -= self.return_road_used
        constraint_4 = self.reward_range[3](self.current_step)
        if output_queue.empty():
            d_constraint_1 = constraint_1 * 1 - self.last_constraint_1
            d_constraint_2 = constraint_2 * 0.3 - self.last_constraint_2
            d_constraint_3 = constraint_3 * 0.2 - self.last_constraint_3
            d_constraint_4 = constraint_4 * 0.1 - self.last_constraint_4
            self.last_constraint_1 = constraint_1 * 1
            self.last_constraint_2 = constraint_2 * 0.3
            self.last_constraint_3 = constraint_3 * 0.2
            self.last_constraint_4 = constraint_4 * 0.1
            reward = d_constraint_1 + d_constraint_2 + d_constraint_3 + d_constraint_4
            return reward
        constraint_1_tag2 = 0
        constraint_2_tag = 1 - output_queue.top().drive
        constraint_2_queue = []

        while not output_queue.empty():
            car = output_queue.get()
            type = car.type
            power = car.power
            drive = car.drive
            number = car.number
            if power == 1:
                if constraint_1_tag != 2 and constraint_1_tag2:
                    constraint_1 -= 1
                constraint_1_tag2 = 1
                constraint_1_tag = 0
            elif constraint_1_tag2:
                constraint_1_tag += 1
            if drive != constraint_2_tag:
                constraint_2_tag = drive
                constraint_2_queue.append(num)
            num += 1

        if len(constraint_2_queue) % 2 == 0:
            if len(constraint_2_queue) == 0:
                pass
            else:
                for i in range(0, len(constraint_2_queue), 2):
                    d1 = constraint_2_queue[i + 1] - constraint_2_queue[i]
                    if i + 2 == len(constraint_2_queue):
                        d2 = num - constraint_2_queue[i + 1]
                    else:
                        d2 = constraint_2_queue[i + 2] - constraint_2_queue[i + 1]
                    if (d2 != d1):
                        constraint_2 -= 1
        else:
            if len(constraint_2_queue) - 1 == 0:
                constraint_2 -= 1
            else:
                constraint_2 -= 1
                for i in range(0, len(constraint_2_queue[:-1]), 2):
                    d1 = constraint_2_queue[i + 1] - constraint_2_queue[i]
                    if i + 2 == len(constraint_2_queue):
                        d2 = num - constraint_2_queue[i + 1]
                    else:
                        d2 = constraint_2_queue[i + 2] - constraint_2_queue[i + 1]
                    if (d2 != d1):
                        constraint_2 -= 1
        d_constraint_1 = constraint_1 * 1 - self.last_constraint_1
        d_constraint_2 = constraint_2 * 0.3 - self.last_constraint_2
        d_constraint_3 = constraint_3 * 0.2 - self.last_constraint_3
        d_constraint_4 = constraint_4 * 0.1 - self.last_constraint_4

        self.last_constraint_1 = constraint_1 * 1
        self.last_constraint_2 = constraint_2 * 0.3
        self.last_constraint_3 = constraint_3 * 0.2
        self.last_constraint_4 = constraint_4 * 0.1
        if done:
            reward = d_constraint_1 + d_constraint_2 + d_constraint_3 + d_constraint_4
            self.reward_result.append(
                [self.last_constraint_1 * 0.4, self.last_constraint_2 * 0.3, self.last_constraint_3 * 0.2,
                 self.last_constraint_4 * 0.1,
                 self.last_constraint_1 * 0.4 + self.last_constraint_2 * 0.3 + self.last_constraint_3 * 0.2 + self.last_constraint_4 * 0.1])
            print(self.reward_result)
            x = np.arange(0, len(self.reward_result), 1)
            y = np.array(self.reward_result).astype(np.float32)
            fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=100)
            plt.plot(x, y[:, 0], label="惩罚一")
            plt.plot(x, y[:, 1], label="惩罚二")
            plt.plot(x, y[:, 2], label="惩罚三")
            plt.plot(x, y[:, 3], label="惩罚四")
            plt.plot(x, y[:, 4], label="总惩罚")
            plt.legend()
            plt.xlabel("调度完成次数")
            plt.ylabel('得分')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            plt.savefig(f"reward_{self.reset_number}.png")
            plt.close('all')
        else:
            reward = 0.4 * d_constraint_1 + 0.3 * d_constraint_2 + 0.2 * d_constraint_3 + 0.1 * d_constraint_4
        return reward

    def step(self, action):  # TODO: 返回 (observation, reward, done, choose_action)
        # TODO: action, (2,)
        result = self._every_car_where()
        self.output_array.append(result)
        self._take_action(action)
        self._every_time_step()
        choose_action = self._get_choose_action()
        self.current_step += 1
        done = self._done()
        reward = self._reward_compute(done=done)
        obs = self._next_observation()
        obs_1, obs_2 = self._key_to_array(obs[0]), obs[1]
        obs_3 = obs[2]
        obs_3_1 = self._queue_to_array(obs_3[0])
        obs_3_2 = self._queue_to_array(obs_3[1])
        obs_time_1, obs_time_2, obs_time_3 = self.observation_require_time, self.agent_observation_require_time, self.tuzhuang_observation_require_time

        obs = [obs_1, obs_2, obs_3_1, obs_3_2, obs_time_1, obs_time_2, obs_time_3]  # TODO: 缓冲区的状态矩阵, 智能体的状态矩阵,
        # TODO: 出装的状态矩阵，总装的状态矩阵,
        # TODO: (缓冲区的状态矩阵剩余时间, 智能体的状态矩阵剩余时间,出装和总装剩余时间)

        if done:
            self.output_array.append(self._every_car_where())
            np.save(f"result_{self.reset_number}.npy", np.stack(self.output_array))
        return obs, reward, done, choose_action

    def _take_action(self, action):  # TODO: 执行action，这里只设置，真正的执行在_every_time_step
        action_1, action_2 = action
        assert -1 <= action_1 < 13 and -1 <= action_2 < 13
        self.agent_observation[0] = self.agent_observation[0] if action_1 == -1 else action_1
        self.agent_observation[1] = self.agent_observation[1] if action_2 == -1 else action_2

    def _get_choose_action(self):  # TODO: 获取可选择的action
        agent_1_choose_action = np.zeros(13, dtype=np.int32)
        agent_2_choose_action = np.zeros(13, dtype=np.int32)

        # agent_1
        if self.agent_observation_require_time[0] - 1 > 0:  # TODO PBS 约束3：接车机上有车时不能采取其他动作
            pass
        else:
            if self.observation[6, 0] != 0 and self.observation_require_time[6, 0] - 1 <= 0:
                # TODO PBS 约束6：当返回道10停车位有车身，同时接车机空闲，优先处理返回道10停车位的车身
                mask = (self.observation[:6, 0] == 0)
                agent_1_choose_action[7:] = mask
                if mask.sum().item() == 0:  # TODO: 存疑点，到底是否需要及时操作
                    agent_1_choose_action[0] = 1
            elif not self.tuzhuang_observation[0].empty():  # TODO: 还有车没从涂装车间出完
                agent_1_choose_action[1:7] = np.where(
                    (self.observation_require_time[:6, 0] == 0) & (self.observation[:6, 0] == 0), 1, 0)
                # TODO: 极端情况 没车位 而且对接车机没约束，没说有车一定需要操作
                if agent_1_choose_action[1:7].sum().item() == 0:
                    agent_1_choose_action[0] = 1
            else:  # TODO: 已经没车了
                agent_1_choose_action[0] = 1

        # agent_2
        if self.agent_observation_require_time[1] - 1 > 0:
            pass
        else:
            car_mask = self.observation[:6, 9] > 0
            require_time_mask = self.observation_require_time[:6, 9] - 1 <= 0
            mask = car_mask & require_time_mask
            if mask.sum().item():  # TODO 约束7 8  先到先得
                reach_time = np.min(self.time_cnt[mask])
                idx = np.argwhere((self.time_cnt == reach_time) & mask) + 1  # TODO: +1
                agent_2_choose_action[idx] = 1
                agent_2_choose_action[idx + 6] = 1
                mask2 = self.observation_require_time[6, 9] > 0
                if mask2:
                    agent_2_choose_action[7:13] = 0
            else:
                agent_2_choose_action[0] = 1
        return [agent_1_choose_action, agent_2_choose_action]

# env = EnvironmentPBS(1000, "/home/sst/product/RL/Datas/data1.xlsx")
# env.observation = (np.random.rand(7, 10) * 2).astype(np.int)
# env.observation_require_time = (np.random.rand(7, 10) * 9).astype(np.int)
# env.observation_require_time = np.where(env.observation > 0, env.observation_require_time, 0)
# print(env.observation)
# for i in range(1000):
#     o = env.observation
#     t = env.observation_require_time
#     env._every_time_step()
#     print(f"step:{i}", "observation:", o, "observation require time:", t)
