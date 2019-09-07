from .us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np
import random
import math

_LOGGER = logging.getLogger(__name__)


class PlaneTaskUsEnv(PhantomUsEnv):
    def __init__(
            self,
            angle_range=None,
            dx_reward_coeff=1,
            angle_reward_coeff=1,
            probe_dislocation_prob=0,
            max_probe_disloc=1,
            max_probe_disrot=1,
            dislocation_seed=42,
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
        self.angle_range = angle_range
        self.angle_reward_coeff = angle_reward_coeff
        self.dx_reward_coeff = dx_reward_coeff
        self.max_probe_disrot = max_probe_disrot
        self.max_probe_disloc = max_probe_disloc
        self.probe_dislocation_prob = probe_dislocation_prob
        self.action_space = spaces.Discrete(len(self._get_action_map()))
        if dislocation_seed:
            self.dislocation_rng = random.Random(dislocation_seed)
        else:
            self.dislocation_rng = None

    def _get_action_map(self):
        return {
            # x, y, theta
            0: (0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0),
            2: (self.step_size,  0, 0),
            3: (0, 0, -self.rot_deg),
            4: (0, 0, self.rot_deg),
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
            3: "ROT_C",
            4: "ROT_CC",
        }.get(action_number, None)

    def _perform_action(self, action):
        x_t, _, theta_t = self._get_action(action)
        z_t = 0
        _LOGGER.debug("Executing action: %s" % str((x_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, z_t)
        p = self.probe.rotate(theta_t)
        if self.angle_range is None or self._is_in_angle_range(p.angle):
            self.probe = p


    def _get_ox_l1_distance(self):
        # use only OX distance
        tracked_pos = self.phantom.get_main_object().belly.pos
        current_pos = self.probe.get_focal_point_pos()
        dx = np.abs(tracked_pos[0]-current_pos[0])
        av_x_l, av_x_r = self._get_available_x_pos()
        # WARN: assuming that the tracked object is a static object (does
        # not move)
        max_dx = max(abs(tracked_pos[0]-av_x_l), abs(tracked_pos[0]-av_x_r))
        return dx/max_dx

    def _get_reward(self):
        # use only OX distance
        distance = self._get_ox_l1_distance() # \in [0,1], 0 is better
        probe_angle = self.probe.angle % 360
        tracked_angle = self.phantom.get_main_object().angle % 360
        angle_diff_sin = math.sin(math.radians(probe_angle - tracked_angle))
        # \in [0,1], 0 is better
        # why not just use difference between angles?
        # "Convert" to reward.
        dist_reward = -distance
        angle_reward = -abs(angle_diff_sin)
        return self.dx_reward_coeff*dist_reward + self.angle_reward_coeff*angle_reward

    def _update_state(self):
        if self.dislocation_rng and \
           self.dislocation_rng.random() < self.probe_dislocation_prob:
            # draw, whether we rotate or accidentally move the probe
            if self.dislocation_rng.random() < 0.5:
                # Dislocate probe on along OX axis.
                x_disloc = self.dislocation_rng.choice(
                    list(range(-self.max_probe_disloc, 0)) +
                    list(range(1, self.max_probe_disloc+1)))
                x_disloc *= self.step_size
                self._move_focal_point_if_possible(x_t=x_disloc, z_t=0)
            else:
                disrot = self.dislocation_rng.choice(
                    list(range(-self.max_probe_disrot, 0)) +
                    list(range(1, self.max_probe_disrot+1)))
                disrot *= self.rot_deg
                self.probe.rotate(disrot)

    def _is_in_angle_range(self, angle):
        left, right = self.angle_range
        return left <= angle <= right

