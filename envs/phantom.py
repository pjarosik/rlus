import numpy as np
import math
import random
from .utils import copy_and_apply, to_string


class Ball:
    """
    The Ball. Object's of this class are immutable.
    """
    def __init__(self, pos, r):
        self.pos = pos
        self.r = r

    def contains(self, points):
        """
        Returns: mask with 1's for points is inside the ball, 0 otherwise.
        """
        return np.sum(np.power(points-self.pos, 2), axis=1) < self.r*self.r

    def translate(self, t):
        """
        Moves the ball by a given vector.

        Args:
            t: a translation vector

        Returns:
            translated ball (a copy).
        """
        return copy_and_apply(self, deep=False, pos=self.pos + t)

    def rotate_xz(self, angle, axis_pos):
        """
        Rotates the position of the ball's center by a given angle (in degrees)
        around given axis. Object is rotated in OXZ plane.

        Args:
            angle: rotation angle, in degrees
            axis_pos: the position of the axis of rotation (x, y, z)

        Returns:
            rotated copy of this object
        """
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        axis_pos = np.array([axis_pos[0], 0, axis_pos[2]])
        x, y, z = self.pos-axis_pos
        new_pos = np.array([c*x-s*z, y, s*x+c*z])+axis_pos
        return copy_and_apply(self, deep=False, pos=new_pos)

    def rotate_xy(self, angle, axis_pos):
        """
        Rotates the position of the ball's center by a given angle (in degrees)
        around given axis. Object is rotated in OXY plane.

        Args:
            angle: rotation angle, in degrees
            axis_pos: the position of the axis of rotation (x, y, z)

        Returns:
            rotated copy of the ball
        """
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        axis_pos = np.array([axis_pos[0], axis_pos[1], 0])
        x, y, z = self.pos-axis_pos
        new_pos = np.array([c*x-s*y, s*x+c*y, z])+axis_pos
        return copy_and_apply(self, deep=False, pos=new_pos)

    def plot_mesh(self, ax, color='r'):
        """
        Visualizes this ball on a given matplotlib's axis.

        Args:
            ax: ax object
            color: the colour of the ball
        """
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.r * np.outer(np.cos(u), np.sin(v))+self.pos[0]
        y = self.r * np.outer(np.sin(u), np.sin(v))+self.pos[1]
        z = self.r * np.outer(np.ones(np.size(u)), np.cos(v))+self.pos[2]
        ax.plot_surface(x, y, z, color=color)

    def __str__(self):
        return to_string(self)


class Teddy:
    """
    Experiments Object of Interest (OOI). Instances of this class are
    immutable.
    """
    def __init__(self, belly_pos, scale, head_offset=1):
        """
        OOI constructor.

        Args:
            belly_pos: position of the belly's center
            scale: the belly's radius
            head_offset: offset between Teddy's belly and the head/paw.
        """
        self.angle = 0 # current rot. angle
        # Belly.
        belly_r = scale
        self.belly = Ball(belly_pos, r=belly_r)
        # Head.
        head_r = scale/2
        head_pos = self.belly.pos+[0, 0, -(self.belly.r * head_offset + head_r)]
        self.head = Ball(head_pos, r=head_r)
        # Paws.
        paw_r = scale/3
        paw_pos = self.belly.pos+[0, 0, -(self.belly.r * head_offset + paw_r)]
        self.paws = [
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=45),
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=135),
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=225),
            Ball(pos=paw_pos, r=paw_r).rotate_xz(axis_pos=belly_pos, angle=315)
        ]

    def contains(self, points):
        """
        Returns:
            mask with 1's if given point is inside the OOI, 0 otherwise.
        """
        ret = self.belly.contains(points)
        ret = np.logical_or(ret, self.head.contains(points))
        for paw in self.paws:
            ret = np.logical_or(ret, paw.contains(points))
        return ret

    def rotate_xy(self, angle, axis_pos=None):
        """
        Rotates the position of the Teddy's belly center by a given angle
        (in degrees) around given axis. Object is rotated in OXY plane.

        Args:
            angle: rotation angle, in degrees
            axis_pos: the position of the axis of rotation (x, y, z)

        Returns:
            rotated copy of the Teddy
        """
        if axis_pos is None:
            axis_pos = self.belly.pos
        new_belly = self.belly.rotate_xy(angle=angle, axis_pos=axis_pos)
        new_head = self.head.rotate_xy(angle=angle, axis_pos=axis_pos)
        new_paws = []
        for paw in self.paws:
            new_paws.append(paw.rotate_xy(angle=angle, axis_pos=axis_pos))
        return copy_and_apply(
            self, deep=True,
            belly=new_belly, head=new_head, paws=new_paws,
            angle=(self.angle+angle)%360
        )

    def translate(self, t):
        """
        Moves the OOI by a given vector.

        Args:
            t: a translation vector

        Returns:
            translated ball (a copy).
        """
        new_belly = self.belly.translate(t)
        new_head = self.head.translate(t)
        new_paws = []
        for paw in self.paws:
            new_paws.append(paw.translate(t))
        return copy_and_apply(
            self, belly=new_belly, head=new_head, paws=new_paws)

    def plot_mesh(self, ax):
        """
        Visualizes OOI on a given matplotlib's axis.

        Args:
            ax: ax object
        """
        color = 'r'
        self.belly.plot_mesh(ax, 'b')
        self.head.plot_mesh(ax, 'r')
        for paw in self.paws:
            paw.plot_mesh(ax, 'r')

    def get_pos(self):
        return self.belly.pos

    def __str__(self):
        return to_string(self)


class ScatterersPhantom:
    """
    Phantom - a container for examined OOIs.
    """
    def __init__(
        self,
        objects,
        x_border,
        y_border,
        z_border,
        n_scatterers,
        n_bck_scatterers,
        seed=None,
    ):
        """
        Phantom's constructor.

        Args:
            objects: phantom's objects
            x_border: a tuple (x limit left, x limit right)
            y_border: a tuple (y limit left, y limit right)
            z_border: a tuple (z limit left, z limit right)
            n_scatterers: number of scatters within are of the OOI - currently Teddy instance
            n_bck_scatterers: number of background scatters -  outside of the OOI area
        """
        self.objects = objects
        self.initial_objects = objects
        self.current_state = 0 # initial pos
        self.x_border = x_border
        self.y_border = y_border
        self.z_border = z_border
        self.n_scatterers = n_scatterers
        self.n_bck_scatterers = n_bck_scatterers
        self.bck_amp = 1
        self.obj_amp = 10
        self.rng = np.random.RandomState(seed=seed)

    def translate(self, t):
        """
        Moves all objects contained by this phantom by a given vector.

        Args:
            t: a translation vector

        Returns:
            translated phantom (a copy).
        """
        new_objects = []
        for obj in self.objects:
            new_objects.append(obj.translate(t))
        return copy_and_apply(self, deep=True, objects=new_objects)

    def rotate_xy(self, angle):
        """
        Rotates the position of all objects within the phantom by a given angle
        (in degrees) around given axis. Object is rotated in OXY plane.

        Args:
            angle: rotation angle, in degrees
            axis_pos: the position of the axis of rotation (x, y, z)

        Returns:
            rotated copy of the phantom
        """
        new_objects = []
        rot_axis = np.array([0,0,0])
        for obj in self.objects:
            new_objects.append(obj.rotate_xy(angle=angle, axis_pos=rot_axis))

        return copy_and_apply(self, deep=True, objects=new_objects)

    def plot_mesh(self, ax):
        """
        Visualizes phantom's objects on a given matplotlib's axis.
        Args:
            ax: ax object
        """
        for obj in self.objects:
            obj.plot_mesh(ax)

    def get_points(self, window):
        # TODO(pjarosik) this method should generate points only once!
        """
        Returns positions and amps of scatterers scanned by the probe, in the
        format ready to use in Field II simulator.
        Returns:
            (points, amps), where:
                points - an array (N,3) with scatterers positions (x,y,z)
                amps - point amplitude
        """
        # FOV.
        x_size, y_size = window  # [m], [m]
        # We consider here the center of phantom.
        xs = (self.rng.rand(self.n_scatterers, 1)-0.5)*x_size
        ys = (self.rng.rand(self.n_scatterers, 1)-0.5)*y_size
        zs = self.rng.rand(self.n_scatterers, 1)
        zs = zs*(self.z_border[1]-self.z_border[0])+self.z_border[0]
        points = np.concatenate((xs, ys, zs), axis=1)
        points_idx = self.objects[0].contains(points)
        for o in self.objects[1:]:
            points_idx = np.logical_or(points_idx, o.contains(points))
        # Reduce the number of background point (to decrease computation time)
        objects_points = points[points_idx]
        background_points = points[np.logical_not(points_idx)]
        background_sample = random.sample(
            list(range(background_points.shape[0])),
            min(self.n_bck_scatterers, background_points.shape[0]))
        background_points = background_points[background_sample]
        points = np.concatenate((objects_points, background_points))
        points_idx = np.array(
            [True]*objects_points.shape[0]+[False]*background_points.shape[0])
        amps = self.rng.randn(points.shape[0], 1)
        amps[points_idx] = self.obj_amp*amps[points_idx]
        bck_amps_idx = np.logical_not(points_idx)
        amps[bck_amps_idx] = self.bck_amp*amps[bck_amps_idx]
        return points, amps

    def get_main_object(self):
        return self.objects[0]

    def __str__(self):
        return to_string(self)


