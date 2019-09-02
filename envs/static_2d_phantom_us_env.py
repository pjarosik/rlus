from .us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np

_LOGGER = logging.getLogger(__name__)


class Static2DPhantomUsEnv(PhantomUsEnv):
    _STEP_SIZE = 10/1000  # [m]
    _ROT_DEG = 10  # [degrees]

    ACTION_NAME_DICT = {
        0: "NOP",
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "DOWN",
    }

    def __init__(self, dx_reward_coeff, dz_reward_coeff, **kwargs):
        super().__init__(**kwargs)
        self.dx_reward_coeff = dx_reward_coeff
        self.dz_reward_coeff = dz_reward_coeff
        self.action_space = spaces.Discrete(len(self._get_action_map()))

    def _get_action_map(self):
        return {
            0: (0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0),  # move to the left
            2: (self.step_size,  0, 0),  # move to the right
            3: (0, -self.step_size, 0),  # move upwards
            4: (0,  self.step_size, 0),  # move downwards
        }

    def get_action_name(self, action_number):
        """
        Returns string representation for given action number
        (e.g. when logging trajectory to file)
        """
        return {
            0: "NOP",
            1: "LEFT",
            2: "RIGHT",
            3: "UP",
            4: "DOWN",
        }.get(action_number, None)

    def _perform_action(self, action):
        x_t, z_t, theta_t = self._get_action(action)
        _LOGGER.debug("Moving the probe: %s" % str((x_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, z_t)

    def _get_reward(self):
        tracked_pos = self.phantom.get_main_object().belly.pos*10 # [mm]
        current_pos = np.array([self.probe.pos[0], 0, self.probe.focal_depth])*10
        dx = np.abs(tracked_pos[0]-current_pos[0])
        dz = np.abs(tracked_pos[2]-current_pos[2])
        reward = -(self.dx_reward_coeff * dx + self.dz_reward_coeff * dz)
        return reward
