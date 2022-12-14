ENV_ID = "CartPole-v1"
# ENV_ID = "LunarLander-v2"
TOTAL_TIMESTEPS = 50000
# TOTAL_TIMESTEPS = 150000
# TOTAL_TIMESTEPS = 250000
SEED = 0

ALGORITHM = "PPO"
# ALGORITHM = "DQN"
POLICY = "MlpPolicy"

USE_CUSTOM_ENV = False
DO_LEARNING = True
DO_EVALUATION = True
