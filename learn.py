import gym
from gym.envs.registration import register
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import const
from customenv.CustomCartPole import CustomCartPole


def main():
    env = get_env(const.ENV_ID)
    env = DummyVecEnv([lambda: env])

    model_file = const.ENV_ID + "_" + const.ALGORITHM + str(const.TOTAL_TIMESTEPS)
    if const.USE_CUSTOM_ENV:
        model_file += "_custom"

    if const.DO_LEARNING:
        model = get_model(env)
        model.learn(total_timesteps=const.TOTAL_TIMESTEPS)
        model.save(model_file)
        del model  # remove to demonstrate saving and loading

    if const.DO_EVALUATION:
        model = get_modle_by_file(model_file)

        try:
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
        except KeyboardInterrupt as e:
            print("detect keyboard interrupt!!")
        finally:
            env.close()


def get_env(env_id: str):
    env = gym.make(env_id)
    if const.USE_CUSTOM_ENV:
        env = CustomCartPole(env)
    return env


def get_model(env):
    if const.ALGORITHM == "PPO":
        model = PPO(
            const.POLICY, env, verbose=1, seed=const.SEED, tensorboard_log="./tlogs/"
        )
    elif const.ALGORITHM == "DQN":
        model = DQN(
            const.POLICY, env, verbose=1, seed=const.SEED, tensorboard_log="./tlogs/"
        )
    else:
        raise ValueError("Not suppoted algorithm.")
    return model


def get_modle_by_file(model_file: str):
    if const.ALGORITHM == "PPO":
        model = PPO.load(model_file)
    elif const.ALGORITHM == "DQN":
        model = DQN.load(model_file)
    else:
        raise ValueError("Not suppoted algorithm.")
    return model


if __name__ == "__main__":
    main()
