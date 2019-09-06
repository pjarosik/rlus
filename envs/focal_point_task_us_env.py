from .us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np
import random

_LOGGER = logging.getLogger(__name__)


class FocalPointTaskUsEnv(PhantomUsEnv):
    def __init__(
            self,
            dx_reward_coeff,
            dz_reward_coeff,
            probe_dislocation_prob=0,
            max_probe_dislocation=2,
            dislocation_seed=None,
            **kwargs):
        """
        Args:
            dx_reward_coeff: L1 dx multiplier
            dz_reward_coeff: L1 dz multiplier
            probe_dislocation_prob: the probability, that probe will be randomly
            dislocated in given timestep
            max_probe_dislocation: maximum random probe dislocation, that can
            be performed, in the number of self.step_sizes
        """
        super().__init__(**kwargs)
        self.max_probe_disloc = max_probe_dislocation
        self.probe_dislocation_prob = probe_dislocation_prob
        self.dx_reward_coeff = dx_reward_coeff
        self.dz_reward_coeff = dz_reward_coeff
        self.action_space = spaces.Discrete(len(self._get_action_map()))
        if dislocation_seed:
            self.dislocation_rng = random.Random(dislocation_seed)
        else:
            self.dislocation_rng = None

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
        tracked_pos = self.phantom.get_main_object().belly.pos
        current_pos = np.array([self.probe.pos[0], 0, self.probe.focal_depth])
        dx = np.abs(tracked_pos[0]-current_pos[0])
        dz = np.abs(tracked_pos[2]-current_pos[2])

        av_x_l, av_x_r = self._get_available_x_pos()
        av_z_l, av_z_r = self._get_available_z_pos()
        max_dx = max(abs(tracked_pos[0]-av_x_l), abs(tracked_pos[0]-av_x_r))
        max_dz = max(abs(tracked_pos[2]-av_z_l), abs(tracked_pos[2]-av_z_r))
        dx = dx/max_dx
        dz = dz/max_dz
        reward = -(self.dx_reward_coeff * dx + self.dz_reward_coeff * dz)
        return reward

    def _update_state(self):
        if self.dislocation_rng and \
           self.dislocation_rng.random() < self.probe_dislocation_prob:
            # draw a random dislocation
            x_disloc = self.dislocation_rng.choice(
                list(range(-self.max_probe_disloc, 0)) +
                list(range(1, self.max_probe_disloc+1)))
            x_disloc *= self.step_size
            self._move_focal_point_if_possible(x_t=x_disloc, z_t=0)
