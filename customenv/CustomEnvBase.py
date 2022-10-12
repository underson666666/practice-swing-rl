import csv
import os
from datetime import datetime

import const
import gym


class CustomEnvBase(gym.Wrapper):
    def init_log(self):
        self.log_dir = os.getcwd()
        now = datetime.now()
        now_str = now.strftime("%Y%m%d%H%M%S")
        self.log_file_name = (
            "logs_"
            + const.ENV_ID
            + "_"
            + const.ALGORITHM
            + "_"
            + str(const.TOTAL_TIMESTEPS)
            + "_"
            + now_str
            + ".csv"
        )
        self.log_path = os.path.join(self.log_dir, self.log_file_name)

        # you have to initialize log header
        # self.log_header = [["update", "step", "r0", "r1", "reward"]]
        # self.append_csv(self.log_header)

        self.log_datas = list()

    def append_csv(self, datas):
        with open(self.log_path, "a") as f:
            w = csv.writer(f)
            w.writerows(datas)
        return

    def create_csv(self):
        self.append_csv(self.log_header)

    def close(self):
        self.env.close()
        self.append_csv(self.log_datas)
        self.log_datas = list()

    def seed(self, seed=None):
        self.env.seed(seed)
