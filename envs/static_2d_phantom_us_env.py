from .us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np

_LOGGER = logging.getLogger(__name__)


class Static2DPhantomUsEnv(PhantomUsEnv):
    _STEP_SIZE = 10/1000  # [m]
    _ROT_DEG = 10  # [degrees]

    _ACTION_DICT = {
        0: (0, 0, 0),  # NOP
        1: (-_STEP_SIZE, 0, 0),  # move to the left
        2: (_STEP_SIZE,  0, 0),  # move to the right
        3: (0, -_STEP_SIZE, 0),  # move upwards
        4: (0,  _STEP_SIZE, 0),  # move downwards
    }
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
        self.action_space = spaces.Discrete(
            len(Static2DPhantomUsEnv._ACTION_DICT))

    def _perform_action(self, action):
        x_t, z_t, theta_t = Static2DPhantomUsEnv._ACTION_DICT[action]
        _LOGGER.debug("Moving the probe: %s" % str((x_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, z_t)

    def get_action_name(self, action):
        """
        Returns string representation for given action number
        (e.g. when logging trajectory to file)
        """
        return Static2DPhantomUsEnv.ACTION_NAME_DICT[action] if action is not None else None

    def _get_reward(self):
        tracked_pos = self.phantom.get_main_object().belly.pos*10 # [mm]
        current_pos = np.array([self.probe.pos[0], 0, self.probe.focal_depth])*10
        dx = np.abs(tracked_pos[0]-current_pos[0])
        dz = np.abs(tracked_pos[2]-current_pos[2])
        reward = -(self.dx_reward_coeff * dx + self.dz_reward_coeff * dz)
        return reward
