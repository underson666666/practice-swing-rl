import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import const
from customenv.CustomCartPole import CustomCartPole

env = gym.make(const.ENV_ID)
env = CustomCartPole(env)
if const.USE_CUSTOM_ENV:
    model_file = "ppo_cartpole_custom"
else:
    model_file = "ppo_cartpole"

if const.DO_LEARNING:
    model = PPO("MlpPolicy", env, verbose=1, seed=const.SEED)
    model.learn(total_timesteps=const.TOTAL_TIMESTEPS)
    model.save(model_file)
    del model  # remove to demonstrate saving and loading

if const.DO_EVALUATION:
    model = PPO.load(model_file)

    while True:
        obs = env.reset()
        dones = False
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            # print(obs)
            # print(rewards)
            # print(dones)
            # print(info)
            break

env.close()
