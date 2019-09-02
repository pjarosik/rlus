import numpy as np
from envs.static_2d_phantom_us_env import Static2DPhantomUsEnv
from envs.phantom import (
    ScatterersPhantom,
    Ball,
    Teddy
)
from envs.imaging import ImagingSystem, Probe
from envs.generator import (
    ConstPhantomGenerator,
    ConstProbeGenerator,
    RandomProbeGenerator)
import envs.logger
import sys

N_STEPS_PER_EPISODE = 32
N_WORKERS = 4


def env_fn(trajectory_logger, probe_generator, phantom_generator=None):
    if not phantom_generator:
        teddy = Teddy(
            belly_pos=np.array([0 / 1000, 0, 50 / 1000]), # X, Y, Z
            scale=12 / 1000,
            head_offset=.9
        )
        phantom = ScatterersPhantom(
            objects=[teddy],
            x_border=(-40 / 1000, 40 / 1000),
            y_border=(-40 / 1000, 40 / 1000),
            z_border=(0, 90 / 1000),
            n_scatterers=int(1e4),
            n_bck_scatterers=int(1e3),
            seed=42,
        )
        phantom_generator = ConstPhantomGenerator(phantom)

    imaging = ImagingSystem(
        c=1540,
        fs=100e6,
        image_width=40 / 1000,
        image_height=90 / 1000,
        image_resolution=(40, 90),  # [pixels]
        median_filter_size=5,
        dr_threshold=-200,
        dec=1,
        no_lines=64
    )
    env = Static2DPhantomUsEnv(
        dx_reward_coeff=2,
        dz_reward_coeff=1,
        imaging=imaging,
        phantom_generator=phantom_generator,
        probe_generator=probe_generator,
        max_steps=N_STEPS_PER_EPISODE,
        no_workers=N_WORKERS,
        use_cache=True,
        trajectory_logger=trajectory_logger,
    )
    return env


def test_reset():
    """Test created to check a single observation/env state visualization."""

    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=10,
        log_state_csv_freq=10,
        log_state_render_freq=10
    )
    probe = Probe(
        pos=np.array([0 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=30 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)
    env = env_fn(trajactory_logger, probe_generator=probe_generator)
    env.reset()


def test_moving_probe_works():
    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=10,
        log_state_csv_freq=10,
        log_state_render_freq=10
    )
    probe = Probe(
        pos=np.array([0 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=30 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)

    env = env_fn(trajactory_logger, probe_generator=probe_generator)
    env.reset()
    env.step(1) # left
    env.step(1) # left
    env.step(2) # right (should come from cache)
    env.step(2) # right (should come from cache)
    env.step(2) # right
    env.step(4) # down
    env.step(3) # up (cached)
    env.step(3) # up


def test_rewards():
    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=10,
        log_state_csv_freq=10,
        log_state_render_freq=10
    )
    probe = Probe(
        pos=np.array([0 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=30 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)

    env = env_fn(trajactory_logger, probe_generator=probe_generator)
    env.reset()
    env.step(1) # left
    env.step(2) # right
    env.step(4) # down
    env.step(3) # up


def test_nop():
    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=10,
        log_state_csv_freq=10,
        log_state_render_freq=10
    )
    probe = Probe(
        pos=np.array([0 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=30 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)

    env = env_fn(trajactory_logger, probe_generator=probe_generator)
    env.reset()
    env.step(0) # NOP
    env.step(2) # right
    env.step(0) # NOP


def test_cannot_move_probe_outside_phantom_area():
    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=10,
        log_state_csv_freq=10,
        log_state_render_freq=10
    )
    probe = Probe(
        pos=np.array([-20 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=10 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)

    env = env_fn(trajactory_logger, probe_generator=probe_generator)
    env.reset()
    env.step(1) # left - BUMP
    env.step(2) # right # -10
    env.step(2) # right # 0
    env.step(2) # right # 10
    env.step(2) # right # 20
    env.step(2) # right # 20 - BUMP
    env.step(3) # up # 0
    env.step(3) # up # 0 - BUMP
    env.step(4) # down # 10
    env.step(4) # down # 20
    env.step(4) # down # 30
    env.step(4) # down # 40
    env.step(4) # down # 50
    env.step(4) # down # 60
    env.step(4) # down # 70
    env.step(4) # down # 80
    env.step(4) # down # 90
    env.step(4) # down # 90 - BUMP


def test_caching_works():
    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=10,
        log_state_csv_freq=10,
        log_state_render_freq=10
    )
    probe = Probe(
        pos=np.array([0 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=30 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)

    env = env_fn(trajactory_logger, probe_generator=probe_generator)
    env.reset()
    env.step(1) # left
    env.step(2) # right (should come from cache)


def test_random_probe_generator():
    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=10,
        log_state_csv_freq=10,
        log_state_render_freq=10
    )

    probe = Probe(
        pos=np.array([0 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=30 / 1000
    )
    teddy = Teddy(
        belly_pos=np.array([0 / 1000, 0, 50 / 1000]), # X, Y, Z
        scale=12 / 1000,
        head_offset=.9
    )
    phantom = ScatterersPhantom(
            objects=[teddy],
            x_border=(-40 / 1000, 40 / 1000),
            y_border=(-40 / 1000, 40 / 1000),
            z_border=(0, 90 / 1000),
            n_scatterers=int(1e4),
            n_bck_scatterers=int(1e3),
            seed=42,
        )
    phantom_generator = ConstPhantomGenerator(phantom)

    probe_generator = RandomProbeGenerator(
            ref_probe=probe,
            object_to_align=teddy,
            seed=42,
            # x_pos default
            # focal_pos default
        )
    env = env_fn(trajactory_logger, probe_generator=probe_generator,
                 phantom_generator=phantom_generator)
    env.reset()
    env.step(1) # left
    env.reset()
    env.step(2)
    env.reset()
    env.step(3)
    env.reset()
    env.step(3)
    env.reset()
    env.step(1)
    env.reset()
    env.step(1)


def test_deep_focus():
    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=sys.argv[1],
        log_action_csv_freq=1,
        log_state_csv_freq=1,
        log_state_render_freq=1
    )
    probe = Probe(
        pos=np.array([0 / 1000, 0, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=0 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)

    env = env_fn(trajactory_logger, probe_generator=probe_generator)
    env.reset()
    env.step(4)  # down - 10
    env.step(4)  # 20
    env.step(4)  # 30
    env.step(4)  # 40
    env.step(4)  # 50
    env.step(4)  # 60
    env.step(4)  # 70
    env.step(4)  # 80
    env.step(4)  # 90


if __name__ == "__main__":
    globals()[sys.argv[1]]()


