import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from tqdm import trange
from flygym.util.config import all_leg_dofs
from scipy.signal import medfilt
from scipy.integrate import ode
import PIL.Image
from ipywidgets import Video
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import mediapy
from gymnasium import spaces
from gym import spaces


class MyNMF(gym.Env):
    def __init__(self, **kwargs):
        self.nmf = NeuroMechFlyMuJoCo(**kwargs)
        num_dofs = len(self.nmf.actuated_joints)
        bound = 0.5
        self.action_space = spaces.Box(low=-bound, high=bound,
                                       shape=(num_dofs,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(num_dofs,))

    def _parse_obs(self, raw_obs):
        features = [
            raw_obs['joints'][0, :].flatten(),
            # raw_obs['fly'].flatten(),
            # what else would you like to include?
        ]
        return np.concatenate(features, dtype=np.float32)

    def reset(self):
        raw_obs, info = self.nmf.reset()
        return self._parse_obs(raw_obs), info

    def step(self, action):
        raw_obs, info = self.nmf.step({'joints': action})
        obs = self._parse_obs(raw_obs)
        joint_pos = raw_obs['joints'][0, :]
        fly_pos = raw_obs['fly'][0, :]
        reward = np.linalg.norm(fly_pos - raw_obs['fly'][-1, :])  # what is your reward function?
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.nmf.render()

    def close(self):
        return self.nmf.close()

    def _reward_lr_course(self):
        """ Implementation of the reward function """
        # reward facing forwards
        heading_reward = np.sum(
            np.cos(self.robot.GetBaseOrientationRollPitchYaw())) * self._time_step * self._action_repeat

        # penalize crouching
        crouching_penalty = -abs(
            self.robot.GetBasePosition()[2] - self.robot._GetDefaultInitPosition()[2]) * self._time_step

        # reward forward motion
        current_base_position = self.robot.GetBasePosition()
        forward_reward = current_base_position[0] - self._last_base_position[0]
        max_dist = MAX_FWD_VELOCITY * self._time_step * self._action_repeat
        forward_reward = forward_reward  # but not if it's cheating

        # penalize sideways motion
        sideways_penalty = -np.abs(current_base_position[1] - self._last_base_position[1])


        self._last_base_position = current_base_position
        self._CoT = np.sum(self.robot.GetMotorTorques() * self.robot.GetMotorVelocities()) * self._time_step

        reward = self._heading_weight * (heading_reward + crouching_penalty) + \
                 self._distance_weight * (forward_reward + sideways_penalty)

        return reward


run_time = 0.5
nmf_env_headless = MyNMF(render_mode='headless',
                         timestep=1e-4,
                         init_pose='stretch',
                         actuated_joints= all_leg_dofs)  # which DoFs would you use?
nmf_model = PPO(MlpPolicy, nmf_env_headless, verbose=1)
nmf_model.learn(total_timesteps=100_000, progress_bar=False)
nmf_env_headless.close()


nmf_env_rendered = MyNMF(render_mode='saved',
                         timestep=1e-4,
                         init_pose='stretch',
                         render_config={'playspeed': 0.1,
                                        'camera': 'Animat/camera_left_top'},
                         actuated_joints= all_leg_dofs)
obs, _ = nmf_env_rendered.reset()
obs_list = []
rew_list = []
for i in range(int(run_time / nmf_env_rendered.nmf.timestep)):
    action, _ = nmf_model.predict(obs)
    obs, reward, terminated, truncated, info = nmf_env_rendered.step(action)
    obs_list.append(obs)
    rew_list.append(reward)
    nmf_env_rendered.render()

nmf_env_rendered.nmf.save_video('filename.mp4')
nmf_env_rendered.close()