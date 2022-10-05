import gym
import numpy as np
from gym import spaces


class GoLeftEnv(gym.Env):

    # ---
    # Gymインターフェースに沿った自作環境．
    # エージェントが常に左に進むことを学習する環境．
    # ---

    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10):
        super(GoLeftEnv, self).__init__()
        self.grid_size = grid_size
        self.agent_pos = grid_size - 1  # エージェントの状態を定義．
        n_actions = 2  # 行動は「0:LEFT, 1:RIGHT」の二通り．
        self.action_space = spaces.Discrete(n_actions)
        # 行動空間はDiscrete(2)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(1,), dtype=np.float32
        )
        # 状態空間はBox(low=0,high=9,shape=(1,))

    def reset(self):

        # ---
        # 環境を初期化するメソッド．
        # サイズ(1,)のNumPy配列を返す
        # ---

        self.agent_pos = self.grid_size - 1  # 状態を初期化．最右とする．
        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, action):

        # ---
        # エージェントの行動(0:LEFT/1:RIGHT)を受けて環境を更新するメソッド，
        # 出力は，状態(NumPy配列)，報酬，終了判定，情報の4つ．
        # ---

        if action == self.LEFT:  # 0
            self.agent_pos -= 1  # 左に動く場合は座標が1だけ減る．
        elif action == self.RIGHT:  # 1
            self.agent_pos += 1
        else:
            raise ValueError(
                f"Received invalid action={action}"
                + f" is not part of the action space."
            )

        # 状態を[0,grid_size-1]区間に収める．
        # 座標9にいる時に行動1:RIGHTを選択すると状態空間にない10に移動してしまうため，
        # そのような場合は9から動かないこととする．
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
        done = bool(self.agent_pos == 0)  # 終了判定．左端(0)に着いたら終了とする．
        reward = 1 if self.agent_pos == 0 else 0  # 報酬．左端に着いたら報酬1を受け取る．
        info = {}  # infoはなくても良いので空辞書にしておく．
        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self, mode="console"):

        # ---
        # 環境をコンソール上に描画するメソッド．
        # ---

        if mode != "console":
            raise NotImplementedError()
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass
