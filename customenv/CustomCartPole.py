import csv
import os
from datetime import datetime

import const
import gym


class CustomCartPole(gym.Wrapper):
    def __init__(self, env):
        super(CustomCartPole, self).__init__(env)
        self.step_counter = 0
        self.update_counter = 0
        self.accum_reward_in_update = 0
        self.accum_r0_in_update = 0
        self.accum_r1_in_update = 0

        self.log_dir = os.getcwd()
        # self.log_file_name = "learn_logs_" + str(const.TOTAL_TIMESTEPS) + ".csv"
        now = datetime.now()
        now_str = now.strftime("%Y%m%d%H%M%S")
        self.log_file_name = (
            "learn_logs_" + str(const.TOTAL_TIMESTEPS) + "_" + now_str + ".csv"
        )
        self.log_path = os.path.join(self.log_dir, self.log_file_name)
        self.log_header = [["update", "step", "r0", "r1", "reward"]]
        self.append_csv(self.log_header)
        self.log_datas = list()

    def reset(self):

        self.step_log = [
            self.update_counter,
            self.step_counter,
            self.accum_r0_in_update,
            self.accum_r1_in_update,
            self.accum_reward_in_update,
        ]
        self.log_datas.append(self.step_log)

        if self.update_counter % 100 == 0:
            self.append_csv(self.log_datas)
            self.log_datas = list()

        obs = self.env.reset()
        self.step_counter = 0
        self.update_counter += 1
        self.accum_reward_in_update = 0
        self.accum_r0_in_update = 0
        self.accum_r1_in_update = 0

        self.env.reset()

        return obs

    def step(self, action):
        obs, rewards, dones, info = self.env.step(action)

        self.accum_r0_in_update += rewards

        if const.USE_CUSTOM_ENV:
            r1 = self.center_position_reward(obs)
            # r1 = self.left_position_reward(obs)
            rewards += r1
            self.accum_r1_in_update += r1

        # 可視化csv用データ更新
        self.step_counter += 1
        self.accum_reward_in_update += rewards

        return obs, rewards, dones, info

    def center_position_reward(self, obs):
        """位置が中央なら高得点"""
        position = abs(obs[0])
        max_position = self.env.observation_space.high[0]
        if position > max_position:
            return 0
        return -(1 / max_position) * position + 1

    def left_position_reward(self, obs):
        """右側にいるほど高報酬"""
        position = abs(obs[0])
        if 2 <= position <= 3:
            return 1
        return 0

    def append_csv(self, datas):
        with open(self.log_path, "a") as f:
            w = csv.writer(f)
            w.writerows(datas)
        return

    def create_csv(self):
        self.append_csv(self.log_header)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.append_csv(self.log_datas)
        self.log_datas = list()

    def seed(self, seed=None):
        self.env.seed(seed)
