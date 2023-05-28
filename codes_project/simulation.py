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
from alive_progress import alive_bar


class Simulation():
    def __init__(self,
                 time_step,
                 run_time=1,
                 out_dir = '../CPGs',
                 friction=1.0
                 ):
        self.out_dir = Path(out_dir)
        self.run_time = run_time
        self.physics_config = {
            'joint_stiffness': 2500,
            'friction': (friction, 0.005, 0.0001),
            'gravity': (0, 0, -9.81e5)}
        self.terrain_config = {'fly_pos': (0, 0, 300),
                          'friction': (friction, 0.005, 0.0001)}
        self.num_steps_base = int(run_time / time_step)


    def save_video(self):
