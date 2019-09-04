import math
import numpy as np
import gym
from gym import spaces
from .generator import ConstPhantomGenerator

import envs.fieldii as fieldii
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
import logging

_LOGGER = logging.getLogger(__name__)


class PhantomUsEnv(gym.Env):
    def __init__(
        self,
        imaging,
        phantom_generator,
        probe_generator,
        max_steps=20,
        no_workers=2,
        trajectory_logger=None,
        angle_reward_coeff=0,
        use_cache=False,
        step_size=10/1000, # [m]
        rot_deg=10 # [deg]
    ):
        # validate
        if use_cache and not isinstance(phantom_generator, ConstPhantomGenerator):
            raise ValueError("Cache can be used with %s instances only." %
                             ConstPhantomGenerator.__name__)

        # set
        self.phantom, self.probe = None, None
        self.phantom_generator = phantom_generator
        self.probe_generator = probe_generator
        self.imaging = imaging
        self.max_steps = max_steps
        self.step_size = step_size
        self.rot_deg = rot_deg
        self.current_step = 0
        self.current_episode = -1
        # To reduce the number of calls to FieldII, we store lastly seen
        # observation.
        self.current_observation = None
        self.angle_reward_coeff = angle_reward_coeff

        self.field_session = fieldii.Field2(no_workers=no_workers)
        self.use_cache = use_cache
        if self.use_cache:
            self._cache = {}

        self.action_space = spaces.Discrete(len(self._get_action_map()))
        observation_shape = (
            imaging.image_resolution[1],
            imaging.image_resolution[0],
            1)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=observation_shape,
            dtype=np.uint8)
        self.metadata = {
            'render.modes': ['rgb_array']
        }
        self.trajectory_logger = trajectory_logger
        _LOGGER.debug("Created environment: %s" % repr(self))
        _LOGGER.debug("Action space: %s" % repr(self.action_space))
        _LOGGER.debug("Observations space: %s" % repr(self.observation_space))

    def _get_action_map(self):
        return {
            0: (0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0),  # move to the left
            2: (self.step_size,  0, 0),  # move to the right
            3: (0, -self.step_size, 0),  # move upwards
            4: (0,  self.step_size, 0),  # move downwards
            5: (0, 0, -self.rot_deg),
            6: (0, 0,  self.rot_deg)
        }

    def _get_action(self, action_number):
        # Function to override.
        return self._get_action_map()[action_number]

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
            5: "ROT_C",
            6: "ROT_CC"
        }.get(action_number, None)

    def reset(self):
        """
        Resets the environment and increments current episode counter.

        Returns:
            first observation after the reset
        """
        _LOGGER.debug("Restarting environment.")
        self.phantom = next(self.phantom_generator)
        self.probe = next(self.probe_generator)
        self.current_step = 0
        self.current_episode += 1
        o = self._get_observation()
        self.current_observation = o

        if self.trajectory_logger is not None:
            self.trajectory_logger.restart(self,
                                           episode_nr=self.current_episode)
            self.trajectory_logger.log_state(
                episode=self.current_episode,
                step=self.current_step,
                env=self)
        return o

    def step(self, action):
        """
        Perform action and move environment state to the next timestep.
        Args:
            action: action to perform, see UsPhantomEnv.ACTION_NAME_DICT for
                    more information.
        Returns:
            observation, reward, is episode over?, diagnostic info (currently empty dict)
        """
        # perform action -> compute reward -> _udpdate_state() -> get_observation -> log state
        if self.current_step >= self.max_steps:
            raise RuntimeError("This episode is over, reset the environment.")
        self.current_step += 1
        self._perform_action(action)
        reward = self._get_reward()
        # Update current state independently to the action
        # (for example apply shaking noise to the probe position).
        self._update_state()
        o = self._get_observation()
        self.current_observation = o
        episode_over = self.current_step >= self.max_steps

        if self.trajectory_logger is not None:
            self.trajectory_logger.log_action(
                episode=self.current_episode,
                step=self.current_step,
                action_code=action,
                reward=reward,
                action_name=self.get_action_name(action)
            )
            self.trajectory_logger.log_state(
                episode=self.current_episode,
                step=self.current_step,
                env=self
            )
        return o, reward, episode_over, {}

    def render(self, mode='rgb_array', views=None):
        """
        Renders current state of the environment.

        Args:
            mode: rendering mode (see self.metdata['render.modes'] for supported
                  modes
            views: imaging views, any combination of values: {'env', 'observation'},
                   if views is None, all ['env', 'observation'] modes are used.
        Returns:
            an output of the renderer
        """
        if views is None:
            views = ["env", "observation"]
        if mode == 'rgb_array':
            return self._render_to_array(views)
        else:
            super(PhantomUsEnv).render(mode=mode)

    def seed(self, seed=None):
        # TODO(pjarosik) gather all seeds here
        raise NotImplementedError("NYI (seeds are currently set by generator's "
                                  "constructors).")

    def close(self):
        self.field_session.close()

    def get_state_desc(self):
        return {
            "probe_x": self.probe.pos[0],
            "probe_z": self.probe.focal_depth,
            "probe_angle": self.probe.angle,
            "obj_x": self.phantom.get_main_object().belly.pos[0],
            "obj_z": self.phantom.get_main_object().belly.pos[2],
            "obj_angle": self.phantom.get_main_object().angle
        }

    def _get_reward(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def _update_state(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def _perform_action(self, action):
        x_t, z_t, theta_t = self._get_action(action)
        _LOGGER.debug("Executing action: %s" % str((x_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, z_t)
        self.probe = self.probe.rotate(theta_t)

    def _move_focal_point_if_possible(self, x_t, z_t):
        pr_pos_x_l = (self.probe.pos[0] - self.probe.width/2) + x_t
        pr_pos_x_r = (self.probe.pos[0] + self.probe.width/2) + x_t
        pr_pos_z = self.probe.focal_depth + z_t
        x_border_l, x_border_r = self.phantom.x_border
        z_border_l, z_border_r = self.phantom.z_border

        # TODO consider storing Decimals or mms directly.
        rel_tol = 0.01

        def le(a, b):
            return a < b or math.isclose(a, b, rel_tol=rel_tol)

        def ge(a, b):
            return a > b or math.isclose(a, b, rel_tol=rel_tol)

        if le(x_border_l, pr_pos_x_l) and ge(x_border_r, pr_pos_x_r):
            self.probe = self.probe.translate(np.array([x_t, 0, 0]))
        if le(z_border_l, pr_pos_z) and ge(z_border_r, pr_pos_z):
            self.probe = self.probe.change_focal_depth(z_t)

    def _get_available_x_pos(self):
        x_border = self.phantom.x_border()
        probe_margin = self.probe.width/2
        return x_border[0]+probe_margin, x_border[1]-probe_margin

    def _get_available_z_pos(self):
        return self.phantom.z_border()

    def _get_observation(self):
        if self.use_cache:
            return self._get_cached_observation()
        else:
            return self._get_image()

    def _get_cached_observation(self):
        # Assumes, that objects in the phantom does not move (are 'static').
        state = (
            int(round(self.probe.pos[0], 3)*1e3),
            int(round(self.probe.focal_depth, 3)*1e3),
            int(round(self.probe.angle))
        )
        if state in self._cache:
            _LOGGER.info("Using cached value for probe state (x, z, theta)=%s"
                          % str(state))
        else:
            bmode = self._get_image()
            self._cache[state] = bmode
        return self._cache[state]

    def _get_image(self):
        points, amps, _ = self.probe.get_fov(self.phantom)
        rf_array, t_start = self.field_session.simulate_linear_array(
            points, amps,
            sampling_frequency=self.imaging.fs,
            no_lines=self.imaging.no_lines,
            z_focus=self.probe.focal_depth)
        bmode = self.imaging.image(rf_array)
        bmode = bmode.reshape(bmode.shape+(1,))
        _LOGGER.debug("B-mode image shape: %s" % str(bmode.shape))
        return bmode

    def _render_to_array(self, views):
        fig = plt.figure(figsize=(4, 4), dpi=200)
        title_elements = [
             ("Episode: %d", self.current_episode),
             ("step: %d", self.current_step),
             ("reward: %.2f", self._get_reward())
        ]
        title_elements = [el[0] % el[1] for el in title_elements if el[1] is not None]
        title = ", ".join(title_elements)
        # fig.suptitle(title)

        view_handlers = {
            'env': self._plot_env,
            'observation': self._plot_bmode
        }
        projections = {
            'env': '3d',
            'observation': None # default
        }
        for i, v in enumerate(views):
            projection = projections[v]
            view_handler = view_handlers[v]
            ax = fig.add_subplot(len(views), 1, i+1, projection=projection)
            view_handler(ax)
        fig.canvas.draw()
        plt.tight_layout()
        b = fig.canvas.tostring_rgb()
        width, height = fig.canvas.get_width_height()
        result = np.fromstring(b, dtype=np.uint8).copy().reshape(height, width, 3)
        plt.close(fig)
        return result

    def _plot_bmode(self, ax):
        if self.current_observation is None:
            raise RuntimeError("Please call 'restart' method first.")
        ax.imshow(self.current_observation.squeeze(), cmap='gray')
        ax.set_xlabel("width $(px)$")
        ax.set_ylabel("depth $(px)$")
        return ax

    def _plot_env(self, ax):
        ax.set_xlabel("$X (mm)$")
        ax.set_ylabel("$Y (mm)$")
        ax.set_zlabel("$Z (mm)$")
        ax.set_xlim(self.phantom.x_border)
        ax.set_ylim(self.phantom.y_border)
        ax.set_zlim(self.phantom.z_border)
        x_ticks = ax.get_xticks()

        def mm_formatter_fn(x, pos):
            return "%.0f" % (x * 1000)
        mm_formatter = matplotlib.ticker.FuncFormatter(mm_formatter_fn)
        ax.xaxis.set_major_formatter(mm_formatter)
        ax.yaxis.set_major_formatter(mm_formatter)
        ax.zaxis.set_major_formatter(mm_formatter)
        # ax.view_init(0, azim=(-self.probe.angle))
        ax.invert_zaxis()
        # Plot phantom objects.
        for obj in self.phantom.objects:
            obj.plot_mesh(ax)
        # Plot focal point position (THE GOAL).
        focal_point_x = self.probe.pos[0]
        focal_point_y = self.probe.pos[1]
        focal_point_z = self.probe.focal_depth
        ax.scatter(
            focal_point_x, focal_point_y, focal_point_z,
            s=400, c='yellow', marker='X')
        # Plot probe line.
        probe_x = .5*self.probe.width*math.cos(math.radians(self.probe.angle))
        probe_y = .5*self.probe.width*math.sin(math.radians(self.probe.angle))
        probe_pt_1 = np.array([probe_x, probe_y, 0])+self.probe.pos
        probe_pt_2 = -np.array([probe_x, probe_y, 0])+self.probe.pos
        ax.plot(
            xs=[probe_pt_1[0], probe_pt_2[0]],
            ys=[probe_pt_1[1], probe_pt_2[1]],
            zs=[0,0],
            c='yellow',
            linewidth=5
        )
        return ax




