import math
import numpy as np
import gym
from gym import spaces
import copy
import random
from scipy import signal, interpolate
import envs.fieldii as fieldii
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
# from memory_profiler import profile


class Ball:
    def __init__(self, pos, r):
        self.pos = pos
        self.r = r

    def contains(self, points):
        # for given points, returns boolean array with true value for points,
        # which belongs to this ball
        return np.sum(np.power(points-self.pos, 2), axis=1) < self.r*self.r

    def translate(self, t):
        return Ball(
            pos=self.pos+t,
            r=self.r
        )

    def rotate_xz(self, angle, axis_pos):
        """
        Rotate position the center of the by given angle (degrees), around given axis.
        Object is rotated in OXZ plane.
        """
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        axis_pos = np.array([axis_pos[0], 0, axis_pos[2]])
        x, y, z = self.pos-axis_pos
        new_pos = np.array([c*x-s*z, y, s*x+c*z])+axis_pos
        return Ball(
            pos=new_pos,
            r=self.r
        )

    def rotate_xy(self, angle, axis_pos):
        """
        Rotate position of the center of the by given angle (degrees), around given axis.
        Object is rotated in OXY plane.
        """
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        axis_pos = np.array([axis_pos[0], axis_pos[1], 0])
        x, y, z = self.pos-axis_pos
        new_pos = np.array([c*x-s*y, s*x+c*y, z])+axis_pos
        return Ball(
            pos=new_pos,
            r=self.r
        )

    def plot_mesh(self, ax, color='r'):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.r * np.outer(np.cos(u), np.sin(v))+self.pos[0]
        y = self.r * np.outer(np.sin(u), np.sin(v))+self.pos[1]
        z = self.r * np.outer(np.ones(np.size(u)), np.cos(v))+self.pos[2]
        ax.plot_surface(x, y, z, color=color)

    def to_string(self):
        return "Ball: pos=%s, radius: %f" % (str(self.pos), self.r)


class Teddy:
    def __init__(self, belly_pos, scale, dist_ahead=1):
        self.angle = 0
        belly_r = scale
        head_r = scale/2
        paw_r = scale/3
        # Belly.
        self.belly = Ball(belly_pos, r=belly_r)
        # Head.
        head_pos = self.belly.pos+[0, 0, -(self.belly.r*dist_ahead+head_r)]
        self.head = Ball(head_pos, r=head_r)
        # Paws.
        paw_pos = self.belly.pos+[0, 0, -(self.belly.r*dist_ahead+paw_r)]
        self.paws = [
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=45),
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=135),
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=225),
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=315)
        ]

    def plot_mesh(self, ax):
        color = 'r'
        self.belly.plot_mesh(ax, 'b')
        self.head.plot_mesh(ax, 'r')
        for paw in self.paws:
            paw.plot_mesh(ax, 'r')

    def to_string(self):
        head_belly = "Teddy: angle={4}\nbelly: pos={0}, r={1},\nhead: pos={2}, r={3}".format(
            self.belly.pos,
            self.belly.r,
            self.head.pos,
            self.head.r,
            self.angle
        )
        paws_str = ""
        for paw in self.paws:
            paws_str = paws_str + ("\npaw: pos={0}, r={1}".format(paw.pos, paw.r))
        return head_belly + paws_str

    def contains(self, points):
        """Returns a boolean array, where true means, that points belong to Teddy.
        """
        ret = self.belly.contains(points)
        ret = np.logical_or(ret, self.head.contains(points))
        for paw in self.paws:
            ret = np.logical_or(ret, paw.contains(points))
        return ret

    def rotate_xy(self, angle, axis_pos=None):
        if axis_pos is None:
            axis_pos = self.belly.pos
        new_belly = self.belly.rotate_xy(angle=angle, axis_pos=axis_pos)
        new_head = self.head.rotate_xy(angle=angle, axis_pos=axis_pos)
        new_paws = []
        for paw in self.paws:
            new_paws.append(paw.rotate_xy(angle=angle, axis_pos=axis_pos))
        new_teddy = copy.deepcopy(self)
        new_teddy.belly = new_belly
        new_teddy.head = new_head
        new_teddy.paws = new_paws
        new_teddy.angle = (self.angle+angle)%360
        return new_teddy

    def translate(self, t):
        new_belly = self.belly.translate(t)
        new_head = self.head.translate(t)
        new_paws = []
        for paw in self.paws:
            new_paws.append(paw.translate(t))
        new_teddy = copy.deepcopy(self)
        new_teddy.belly = new_belly
        new_teddy.head = new_head
        new_teddy.paws = new_paws
        return new_teddy


class Probe:
    def __init__(self, pos, angle, width, height, focal_depth):
        self.pos = np.array(pos)
        self.angle = angle
        self.width = width
        self.height = height
        self.focal_depth = focal_depth

    def translate(self, t):
        probe = copy.deepcopy(self)
        probe.pos = self.pos+t
        return probe

    def rotate(self, angle):
        probe = copy.deepcopy(self)
        probe.angle = (probe.angle+angle)%360
        return probe

    def change_focal_depth(self, delta_y):
        probe = copy.deepcopy(self)
        probe.focal_depth += delta_y
        return probe

    def to_string(self):
        return "Probe: pos=%s, angle=%f, focal_depth=%f" % (str(self.pos), self.angle, self.focal_depth)


class Imaging:
    def __init__(
        self,
        c, fs,
        image_width,
        image_height,
        image_resolution,
        median_filter_size,
        no_lines,
        dr_threshold,
        dec=1
    ):
        self.c = c
        self.fs = fs
        self.image_width = image_width
        self.image_height = image_height
        self.image_resolution = image_resolution
        self.median_filter_size = median_filter_size
        self.no_lines = no_lines
        self.dr_threshold = dr_threshold
        self.dec = dec

    def _interp(self, data):
        input_xs = np.arange(0, data.shape[1])*(self.image_width/data.shape[1])
        input_zs = np.arange(0, data.shape[0])*(self.c/(2*self.fs))
        output_xs = np.arange(self.image_width, step=self.image_width/self.image_resolution[0])[:self.image_resolution[0]] # FIXME
        output_zs = np.arange(self.image_height, step=self.image_height/self.image_resolution[1])[:self.image_resolution[1]]
        return interpolate.interp2d(input_xs, input_zs, data, kind="cubic")(output_xs, output_zs)

    def _detect_envelope(self, data):
        return np.abs(signal.hilbert(data, axis=0))

    def _adjust_dynamic_range(self, data, dr=-60):
        nonzero_idx = data != 0
        data = 20*np.log10(np.abs(data)/np.max((np.abs(data[nonzero_idx]))))
        return np.clip(data, dr, 0)

    def create_bmode(self, rf):
        # decimate redundant data
        data = rf[::self.dec, :]
        data = self._detect_envelope(rf)
        data = self._adjust_dynamic_range(data, dr=self.dr_threshold)
        data = self._interp(data)
        data = signal.medfilt(data, kernel_size=5)
        # convert to unit8
        data = data-data.min()
        data = (255*(data/data.max())).astype(np.uint8)
        return data


class Phantom:
    def __init__(
        self,
        objects,
        x_border,
        y_border,
        z_border,
        n_scatterers,
        n_bck_scatterers):
        self.objects = objects
        self.x_border = x_border
        self.y_border = y_border
        self.z_border = z_border
        self.n_scatterers = n_scatterers
        self.n_bck_scatterers = n_bck_scatterers

    def step(self):
        """Moves all objects in phantom according to their velocity."""
        pass

    def translate(self, t):
        """Translates objects according to given translation vector."""
        new_objects = []
        for obj in self.objects:
            new_objects.append(obj.translate(t))
        # FIXME use copy.deepcopy!
        return Phantom(
            objects=new_objects,
            x_border=self.x_border,
            y_border=self.y_border,
            z_border=self.z_border,
            n_scatterers=self.n_scatterers,
            n_bck_scatterers=self.n_bck_scatterers
        )

    def rotate_xy(self, angle):
        """rotates all objects around [0,0,0] of the phantom"""
        new_objects = []
        rot_axis = np.array([0,0,0])
        for obj in self.objects:
            new_objects.append(obj.rotate_xy(angle=angle, axis_pos=rot_axis))
        # FIXME use copy.deepcopy!
        return Phantom(
            objects=new_objects,
            x_border=self.x_border,
            y_border=self.y_border,
            z_border=self.z_border,
            n_scatterers=self.n_scatterers,
            n_bck_scatterers=self.n_bck_scatterers
        )

    def get_points(self, probe):
        """returns points, amplitudes for section covered by probe."""
        p_cpy = self.translate(-probe.pos)
        p_cpy = p_cpy.rotate_xy(-probe.angle)
        points, amps = p_cpy._get_fieldii_points(probe)
        return points, amps, p_cpy

    def _get_fieldii_points(self, probe):
        x_size = probe.width # [m]
        # TODO y_size should be equal to y_border width,
        # but that would increase field2 computation time
        y_size = probe.height # [m]
        # we consider here the center of phantom
        xs = (np.random.rand(self.n_scatterers, 1)-0.5)*x_size
        ys = (np.random.rand(self.n_scatterers, 1)-0.5)*y_size
        zs = np.random.rand(self.n_scatterers, 1)*(self.z_border[1]-self.z_border[0])+self.z_border[0]
        points = np.concatenate((xs, ys, zs), axis=1)
        points_idx = self.objects[0].contains(points)
        for o in self.objects[1:]:
            points_idx = np.logical_or(points_idx, o.contains(points))
        # reduce the number of background points
        objects_points = points[points_idx]
        background_points = points[np.logical_not(points_idx)]
        background_sample = random.sample(list(range(background_points.shape[0])), self.n_bck_scatterers)
        background_points = background_points[background_sample]
        points = np.concatenate((objects_points, background_points))
        points_idx = np.array([True]*objects_points.shape[0]+[False]*background_points.shape[0])
        amps = np.random.randn(points.shape[0], 1)*100
        amps[points_idx] = 1000*amps[points_idx]
        return points, amps

    def plot_mesh(self, ax):
        for obj in self.objects:
            obj.plot_mesh(ax)

    def to_string(self):
        strs = [o.to_string() for o in self.objects]
        return "\n".join(strs)


class UsPhantomEnv(gym.Env):
    _ACTION_DICT = {
        0: (0, 0, 0), # NOP
        1: (-10/1000, 0, 0), # move to left -10 [mm]
        2: (10/1000,  0, 0), # move to right  10
        3: (0, -10/1000, 0), # move to upwards -10
        4: (0,  10/1000, 0), # move to downwards -10
        5: (0, 0, -10), # rotate -10 [degrees]
        6: (0, 0,  10) # rotate 10
    }
    _ACTION_NAME_DICT = {
        0: "NOP", # NOP
        1: "LEFT", # move to left -10 [mm]
        2: "RIGHT", # move to right  10
        3: "UP", # move to upwards -10
        4: "DOWN", # move to downwards -10
        5: "ROT_L", # rotate -10 [degrees]
        6: "ROT_R" # rotate 10
    }

    def __init__(
            self,
            imaging,
            env_generator,
            max_steps=20,
            no_workers=2,
            log_freq=10,
            log_dir=None,
            angle_reward_coeff=100,
            use_cache=False # WARN: currently this can be used with const_generators only!
    ):
        self.env_generator = env_generator
        self.phantom, self.probe = next(self.env_generator)
        self.imaging = imaging
        self.max_steps = max_steps
        self.current_step = 0
        self.current_episode = -1
        self.log_freq = log_freq # TODO prepare external logger
        self.log_dir = log_dir
        self.angle_reward_coeff = angle_reward_coeff
        self.action_space = spaces.Discrete(len(UsPhantomEnv._ACTION_DICT))
        # width, height -> height, width
        observation_shape = (imaging.image_resolution[1], imaging.image_resolution[0], 1)
        self.observation_space = spaces.Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)
        self.field_session = fieldii.Field2(no_workers=no_workers)

        self.use_cache = use_cache
        if self.use_cache:
            assert self.env_generator.__name__ in {const_env_generator.__name__, const_phantom_generator.__name__}, "Cache can be used with const phantom generator only."
            self._cache = {}

    def _get_image(self):
        """Creates b-mode image from current state of env and probe."""
        points, amps, _ = self.phantom.get_points(self.probe)
        rf_array, t_start = self.field_session.simulate_linear_array(
            points, amps,
            sampling_frequency=self.imaging.fs,
            no_lines=self.imaging.no_lines,
            z_focus=self.probe.focal_depth)
        bmode = self.imaging.create_bmode(rf_array)
        bmode = bmode.reshape(bmode.shape+(1,))
        print("Bmode shape %s" % str(bmode.shape))
        return bmode

    def _get_cached_obs(self):
        state = (int(round(self.probe.pos[0], 3)*1e3), int(round(self.probe.focal_depth, 3)*1e3), int(round(self.probe.angle)))
        if state in self._cache:
            print("Using cached value for probe state (x, z, theta)=%s" % str(state))
        else:
            bmode = self._get_image()
            self._cache[state] = bmode
        return self._cache[state]

    def _get_observation(self):
        if self.use_cache:
            return self._get_cached_obs()
        else:
            return self._get_image()


    # @profile
    def step(self, action):
        """
        @param action: a vector [x, z, theta], where
        x: 1 - move the probe to the left, 2 - move to the right, 0 - don't move
        z: 1 - move the focal depth upwards, 2 - move the focal_depth downwards, 0 - don't move
        theta: 1 - rotate probe left, 2 - rotate probe right, 0 - don't move
        """
        if self.current_step >= self.max_steps:
            raise RuntimeError("This episode is over, reset the environment.")
        self.current_step += 1
        # updated state of the phantom and probe
        # TODO self.phantom = self.phantom.step()
        x_t, z_t, theta_t = UsPhantomEnv._ACTION_DICT[action]
        print("Executing action: %s" % str((x_t, z_t, theta_t)))
        # Constraints: do not go outside of the phantom box.
        if (self.probe.pos[0] + x_t) < self.phantom.x_border[1]:
            if (self.probe.pos[0] + x_t) > self.phantom.x_border[0]:
                self.probe = self.probe.translate(np.array([x_t, 0, 0]))

        if (self.probe.focal_depth+z_t) < self.phantom.z_border[1]:
            if (self.probe.focal_depth+z_t) > self.phantom.z_border[0]:
                self.probe = self.probe.change_focal_depth(z_t)

        self.probe = self.probe.rotate(theta_t)
        # create new observation
        ob = self._get_observation()
        # compute reward
        reward = self._compute_reward()
        episode_over = self.current_step >= self.max_steps

        if self.current_episode % self.log_freq == 0:
            self._log(action, ob, reward, self.current_step)

        return ob, reward, episode_over, {}

    def _compute_reward(self):
        tracked_object = [obj for obj in self.phantom.objects if isinstance(obj, Teddy)][0]
        tracked_pos = tracked_object.belly.pos*1e3 # [mm]
        current_pos = np.array([self.probe.pos[0], 0, self.probe.focal_depth]) * 1e3
        reward = -np.sqrt(np.sum(np.square(tracked_pos-current_pos)))
        reward += self.angle_reward_coeff*abs(math.cos(math.radians(self.probe.angle-tracked_object.angle)))
        return reward

    def reset(self):
        print("Restarting environment.")
        self.phantom, self.probe = next(self.env_generator)
        self.current_step = 0
        self.current_episode += 1
        obs = self._get_observation()
        if self.current_episode % self.log_freq == 0:
            reward = self._compute_reward()
            self._log(None, obs, reward, 0)
        return obs


    def render_pyplot(self, ax, mode='human', close=False):
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        ax.set_zlabel("$Z$")
        ax.set_xlim(self.phantom.x_border)
        ax.set_ylim(self.phantom.y_border)
        ax.set_zlim(self.phantom.z_border)
        # print(self.probe.angle)
        # ax.view_init(0, azim=(-self.probe.angle))
        ax.invert_zaxis()
        for obj in self.phantom.objects:
            obj.plot_mesh(ax)
        # probe focal point position (THE GOAL)
        focal_point_x = self.probe.pos[0]
        focal_point_y = self.probe.pos[1]
        focal_point_z = self.probe.focal_depth
        ax.scatter(focal_point_x, focal_point_y, focal_point_z, s=1000, c='yellow', marker='X')
        # plot probe line
        probe_x = .5*self.probe.width*math.cos(math.radians(self.probe.angle))
        probe_y = .5*self.probe.width*math.sin(math.radians(self.probe.angle))
        probe_pt_1 = np.array([probe_x, probe_y, 0])+self.probe.pos
        probe_pt_2 = -np.array([probe_x, probe_y, 0])+self.probe.pos
        ax.plot(
            xs=[probe_pt_1[0], probe_pt_2[0]],
            ys=[probe_pt_1[1], probe_pt_2[1]],
            zs=[0,0],
            c='yellow',
            linewidth=10
        )

    def to_string(self):
        return self.probe.to_string() + "\n" + self.phantom.to_string()

    def __getstate__(self):
        return {
            'phantom' : self.phantom,
            'probe' : self.probe,
            'imaging' : self.imaging,
            'max_steps' : self.max_steps,
            'current_step' : self.current_step,
            'action_space' : self.action_space,
            'observation_space' : self.observation_space,
        }

    def __setstate__(self):
        # TODO implement
        raise NotImplementedError

    def _log(self, action, ob, reward, step):
        if self.log_dir is not None:
            episode_dir = os.path.join(self.log_dir, "episode_%d"%self.current_episode)
            os.makedirs(episode_dir, exist_ok=True)

            # Save log file
            tracked_object = [obj for obj in self.phantom.objects if isinstance(obj, Teddy)][0]
            log_file = os.path.join(episode_dir, "steps.txt")
            log_header = [
                'episode',
                'step',
                'action',
                'reward',
                'probe_x',
                'probe_z',
                'probe_angle',
                'obj_x',
                'obj_z',
                'obj_angle'
            ]
            log_values = [
                self.current_episode,
                step,
                UsPhantomEnv._ACTION_NAME_DICT[action] if action is not None else None,
                reward,
                self.probe.pos[0],
                self.probe.focal_depth,
                self.probe.angle,
                tracked_object.belly.pos[0],
                tracked_object.belly.pos[2],
                tracked_object.angle
            ]
            if not os.path.isfile(log_file):
                # first step, write header
                with open(log_file, 'a') as f:
                    f.write('\t'.join(log_header))
                    f.write('\n')
            # Write log values.
            with open(log_file, 'a') as f:
                log_values = [str(v) for v in log_values]
                f.write('\t'.join(log_values))
                f.write('\n')
            # Write visualization of the env and observation.
            # Visualize observation.
            title_elements = [("ep: %d", self.current_episode), ("step: %d", step), ("reward: %.2f", reward)]
            title_elements = [el[0] % el[1] for el in title_elements if el[1] is not None]
            title = ",".join(title_elements)
            fig = plt.figure()
            plt.title(title)
            plt.imshow(ob.squeeze(), cmap='gray')
            plt.savefig(os.path.join(episode_dir, "step_%03d.png" % step))
            plt.close(fig)
            # Visualize environment.
            fig = plt.figure()
            plt.title(title)
            fig.set_size_inches(10, 10)
            ax = fig.add_subplot(111, projection='3d')
            self.render_pyplot(ax)
            plt.savefig(os.path.join(episode_dir, "env_%03d.png" % step))
            plt.close(fig)
            # Pickle dump phantom and probe.
            with open(os.path.join(episode_dir, 'state_%03d.pkl'% step), "wb") as f:
                pickle.dump({'phantom': self.phantom, 'probe': self.probe}, f)

def const_env_generator(phantom, probe):
    while True:
        yield phantom, probe

def _get_rand(start, end):
        return random.random()*(end-start)+start

def const_phantom_generator(phantom):
    z_border = phantom.z_border
    eps_intersect = 5/1000 # how much probe's FOV and the object intersect
    tracked_object = [obj for obj in phantom.objects if isinstance(obj, Teddy)][0]
    teddy_pos = tracked_object.belly.pos
    teddy_r = tracked_object.belly.r
    probe_width = 40/1000
    eps = 5/1000
    z_upper_eps = 20/1000
    z_range_start, z_range_end = (z_border[0]+teddy_r+z_upper_eps, z_border[1]-teddy_r-eps)
    while True:
        probe_x_range_start = teddy_pos[0]-teddy_r-(probe_width/2)+eps_intersect
        probe_x_range_end = teddy_pos[0]+teddy_r+(probe_width/2)-eps_intersect
        probe_x = get_rand(probe_x_range_start, probe_x_range_end)
        probe_pos = np.array([round(probe_x, 2), 0, 0]) # round to 1[cm]
        focal_range_start = max(40/1000, z_range_start-20/1000)
        focal_range_end = min(z_border[1], z_range_end+20/1000)
        probe_focal_depth = round(get_rand(focal_range_start, focal_range_end), 2)
        probe_angle = round(get_rand(0, 90), -1)
        probe = Probe(
            pos=probe_pos,
            angle=probe_angle,
            width=probe_width,
            height=10/1000,
            focal_depth=probe_focal_depth
        )
        yield phantom, probe


def random_env_generator():
    """Infinite generator of phantoms with randomly located objects and probe."""
    x_border=(-60/1000, 60/1000)
    y_border=(-60/1000, 60/1000)
    z_border=(0,        90/1000)

    teddy_r = 10/1000 # TODO randomize
    eps = 5/1000
    z_upper_eps = 20/1000
    x_range_start, x_range_end = (x_border[0]+teddy_r+eps, x_border[1]-teddy_r-eps)
    z_range_start, z_range_end = (z_border[0]+teddy_r+z_upper_eps, z_border[1]-teddy_r-eps)

    probe_width = 40/1000
    eps_intersect = 5/1000 # how much probe's FOV and the object intersect
    while True:
        # Find position for Teddy.
        teddy_pos = np.array([round(_get_rand(x_range_start, x_range_end), 2), 0, round(_get_rand(z_range_start, z_range_end), 2)]) # round to 1 [cm]
        teddy_angle = round(_get_rand(0, 90), -1)
        # Find position for the probe.
        probe_x_range_start = teddy_pos[0]-teddy_r-(probe_width/2)+eps_intersect
        probe_x_range_end = teddy_pos[0]+teddy_r+(probe_width/2)-eps_intersect
        probe_x = _get_rand(probe_x_range_start, probe_x_range_end)
        probe_pos = np.array([round(probe_x, 2), 0, 0]) # round to 1[cm]
        focal_range_start = max(40/1000, z_range_start-20/1000)
        focal_range_end = min(z_border[1], z_range_end+20/1000)
        probe_focal_depth = round(_get_rand(focal_range_start, focal_range_end), 2)
        probe_angle = round(_get_rand(0, 90), -1)
        probe = Probe(
            pos=probe_pos,
            angle=probe_angle,
            width=probe_width,
            height=10/1000,
            focal_depth=probe_focal_depth
        )
        phantom = Phantom(
            x_border=x_border, y_border=y_border, z_border=z_border,
            objects=[
                Teddy(
                    belly_pos=teddy_pos,
                    scale=teddy_r,
                    dist_ahead=.9
                )
                .rotate_xy(angle=teddy_angle),
            ],
            n_scatterers=int(4e4),
            n_bck_scatterers=int(2e3)
        )
        yield phantom, probe


