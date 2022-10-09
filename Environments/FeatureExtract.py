import numpy as np
import torch
import torch.nn as nn


class FE:
    def __init__(self, queue_number):
        self.queue_number = queue_number

    def __call__(self, obs, reward, done, choose):
        [obs_1, obs_2, obs_3_1, obs_3_2, obs_time_1, obs_time_2, obs_time_3] = obs
        obs_1[..., 0] = ~(obs_1[..., 0] == -1)  # TODO: (7,10,4)
        new_obs_1 = np.stack([obs_1[..., 0], obs_1[..., 2], obs_1[..., 3]], -1)  # TODO: (7, 10, 3)
        new_obs_1 = torch.from_numpy(new_obs_1).float()
        new_obs_2 = torch.from_numpy(obs_2).float()  # TODO: (2,)
        obs_3_1 = obs_3_1[:, 2:]
        b, c = obs_3_1.shape
        tmp = -np.ones((self.queue_number - b, 2))
        new_obs_3_1 = np.concatenate([obs_3_1, tmp], 0)
        obs_3_2 = obs_3_2[:, 2:]
        b, c = obs_3_2.shape
        tmp = -np.ones((self.queue_number - b, 2))
        new_obs_3_2 = np.concatenate([obs_3_2, tmp], 0)
        new_obs_3 = torch.from_numpy(
            np.concatenate([new_obs_3_1, new_obs_3_2], 0)
        ).float()  # TODO: (queue_number * 2, 2 )
        new_obs_time_1 = torch.from_numpy(obs_time_1).float()  # TODO: (7, 10)
        new_obs_time_2 = torch.from_numpy(obs_time_2).float()  # TODO: (2,)
        new_obs_time_3 = torch.from_numpy(obs_time_3).float()  # TODO: (2,)

        choose_action_1, choose_action_2 = (
            torch.from_numpy(choose[0]).float(),
            torch.from_numpy(choose[1]).float(),
        )
        return (
            [new_obs_1, new_obs_2, new_obs_3, new_obs_time_1, new_obs_time_2, new_obs_time_3],
            reward,
            done,
            [choose_action_1, choose_action_2],
        )
