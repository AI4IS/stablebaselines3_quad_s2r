import gym
from quad_sim.quadrotor import QuadrotorEnv
from os import path
from stable_baselines3 import SAC

# def wrap_quad(**kwargs):
#     return QuadrotorEnv(**kwargs)

# # Register the wrapped environment
# gym.register(
#     id="Quadrotor-v1",
#     entry_point=wrap_quad,
#     max_episode_steps=1200
# )

env = gym.make("Quadrotor-v1")

log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')

try:
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_path, gradient_steps=-1, device='cpu')
    model.learn(3000000)
finally:
    model.save("models/quad_sac")
    model.save_replay_buffer("models/quad_sac_buffer")