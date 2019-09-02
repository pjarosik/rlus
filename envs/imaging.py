import numpy as np
import random
from scipy import signal, interpolate
from .utils import to_string, copy_and_apply


class Probe:
    def __init__(self, pos, angle, width, height, focal_depth):
        self.pos = np.array(pos)
        self.angle = angle
        self.width = width  # OX
        self.height = height  # OY
        self.focal_depth = focal_depth

    def translate(self, t):
        """
        Moves the position of the probe.

        Args:
            t: translation vector

        Returns:
            displaced probe (a copy)
        """
        return copy_and_apply(
            self, deep=True,
            pos=self.pos+t
        )

    def rotate(self, angle):
        """
        Rotates scanning plane of the probe.

        Args:
            angle: rotation angle (in degrees).

        Returns:
            rotated probe (a copy)
        """
        return copy_and_apply(
            self, deep=True,
            angle=(self.angle+angle)%360)

    def change_focal_depth(self, delta_y):
        """
        Moves upwards/downwards a focal depth of the imaging system.

        Args:
            delta_y: displacement of the focal point

        Returns:
            a probe with new position of the focal point
        """
        return copy_and_apply(
            self, deep=True,
            focal_depth=self.focal_depth+delta_y)

    def get_focal_point_pos(self):
        """
        Returns:
            a 3-D array with the position of the focal point
        """
        return np.array([self.pos[0], self.pos[1], self.focal_depth])

    def get_fov(self, phantom):
        """
        Returns the Field of View from given position and angle of
        the probe.

        Returns:
            points, amplitudes, phantom in new FOV
        """
        ph_cpy = phantom.translate(-self.pos)
        ph_cpy = ph_cpy.rotate_xy(-self.angle)
        points, amps = ph_cpy.get_points(
            window=(self.width, self.height))
        return points, amps, ph_cpy

    def __str__(self):
        return to_string(self)


class ImagingSystem:
    def __init__(
        self,
        c, fs,
        image_width,
        image_height,
        image_resolution,
        median_filter_size,
        dr_threshold,
        no_lines,
        dec=1
    ):
        """
        ImagingSystem's constructor.

        Args:
            c: speed of sound
            fs: sampling frequency
            image_width: width of the output image, in [m]
            image_height: height of the output image, in [m]
            image_resolution: image resolution, (width, height) [pixels]
            median_filter_size: the size of median filter
            dr_threshold: dynamic range threshold
            dec: RF data decimation factor
        """
        self.c = c
        self.fs = fs
        self.image_width = image_width
        self.image_height = image_height
        self.image_resolution = image_resolution
        self.median_filter_size = median_filter_size
        self.dr_threshold = dr_threshold
        self.dec = dec
        self.no_lines = no_lines

    def _interp(self, data):
        input_xs = np.arange(0, data.shape[1])*(self.image_width/data.shape[1])
        input_zs = np.arange(0, data.shape[0])*(self.c/(2*self.fs))
        output_xs = np.arange(
            self.image_width,
            step=self.image_width/self.image_resolution[0])
        output_zs = np.arange(
            self.image_height,
            step=self.image_height/self.image_resolution[1])
        return interpolate.interp2d(input_xs, input_zs, data, kind="cubic")\
            (output_xs, output_zs)

    def _detect_envelope(self, data):
        return np.abs(signal.hilbert(data, axis=0))

    def _adjust_dynamic_range(self, data, dr=-60):
        nonzero_idx = data != 0
        data = 20*np.log10(np.abs(data)/np.max((np.abs(data[nonzero_idx]))))
        return np.clip(data, dr, 0)

    def image(self, rf):
        """
        Computes new B-mode image from given RF data.

        Args:
            rf: recorded ultrasound signal to image

        Returns:
            B-mode image with values \in [0, 1]
        """
        data = rf[::self.dec, :]
        data = self._detect_envelope(data)
        data = self._adjust_dynamic_range(data, dr=self.dr_threshold)
        data = self._interp(data)
        data = signal.medfilt(data, kernel_size=self.median_filter_size)
        data = data-data.min()
        data = data/data.max()
        return data

