from spinup import vpg
import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete
from envs.focal_point_task_us_env import FocalPointTaskUsEnv
from envs.phantom import (
    ScatterersPhantom,
    Ball,
    Teddy
)
from envs.imaging import ImagingSystem, Probe
from envs.generator import ConstPhantomGenerator, RandomProbeGenerator
import envs.logger
import matplotlib
import argparse

N_STEPS_PER_EPISODE =  16
N_STEPS_PER_EPOCH = 64
EPOCHS = 251 # NO_EPISODES = (NSTEPS_PER_EPOCH/NSTEPS_PER_EPISODE)*EPOCHS
N_WORKERS = 4


def env_fn(trajectory_logger):
    probe = Probe(
        pos=np.array([-20 / 1000, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=10 / 1000
    )
    teddy = Teddy(
        belly_pos=np.array([0 / 1000, 0, 50 / 1000]),
        scale=12 / 1000,
        head_offset=.9
    )
    phantom = ScatterersPhantom(
        objects=[teddy],
        x_border=(-45 / 1000, 45 / 1000),
        y_border=(-45 / 1000, 45 / 1000),
        z_border=(0, 90 / 1000),
        n_scatterers=int(1e4),
        n_bck_scatterers=int(1e3),
        seed=42,
    )
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
    env = FocalPointTaskUsEnv(
        dx_reward_coeff=1,
        dz_reward_coeff=1,
        imaging=imaging,
        phantom_generator=ConstPhantomGenerator(phantom),
        probe_generator=RandomProbeGenerator(
            ref_probe=probe,
            object_to_align=teddy,
            seed=42,
            x_pos=np.arange(-25/1000, 29/1000, step=5/1000),
            focal_pos=[10/1000]
        ),
        max_steps=N_STEPS_PER_EPISODE,
        no_workers=N_WORKERS,
        use_cache=True,
        trajectory_logger=trajectory_logger,
        step_size=5/1000
    )
    return env

AC_KWARGS = dict(
    hidden_sizes=[16, 32],
    activation=tf.nn.relu
)


# Below functions base on openai.spinup's A-C scheme implementation.
def cnn(x,
        training_ph,
        hidden_sizes=(32,),
        kernel_size=(3, 3),
        pool_size=(2, 2),
        output_activation=None
        ):
    x = tf.layers.batch_normalization(x, training=training_ph)
    for h in hidden_sizes[:-1]:
        x = tf.layers.conv2d(x, filters=h, kernel_size=kernel_size)
        x = tf.layers.batch_normalization(x, training=training_ph)
        x = tf.nn.relu(x)
        # x = tf.nn.tanh(x)
        x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=pool_size)
    x = tf.layers.flatten(x)
    return tf.layers.dense(x, units=hidden_sizes[-1],
                           activation=output_activation)


def cnn_categorical_policy(x, a, training_ph, hidden_sizes, output_activation,
                           action_space):
    act_dim = action_space.n
    logits = cnn(x, training_ph, hidden_sizes=list(hidden_sizes) + [act_dim],
                 output_activation=None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits, 1),
                    axis=1)  # action drawn from current policy
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all,
                         axis=1)  # log probability of given actions
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all,
                            axis=1)  # log probability of actions of given pi
    return pi, logp, logp_pi, logp_all


def cnn_actor_critic(x, a, training_ph, hidden_sizes=(64, 64),
                     activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):
    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = cnn_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = cnn_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi, logp_all = policy(x, a, training_ph, hidden_sizes,
                                             output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(
            cnn(x, training_ph, hidden_sizes=list(hidden_sizes) + [1],
                output_activation=None), axis=1)
    return pi, logp, logp_pi, v, logp_all


def main():
    matplotlib.use('agg')

    parser = argparse.ArgumentParser(description="Train agent in env: %s" %
                                                 FocalPointTaskUsEnv.__name__)
    parser.add_argument("--exp_dir", dest="exp_dir",
                        help="Where to put all information about the experiment",
                        required=True)

    args = parser.parse_args()

    trajactory_logger = envs.logger.TrajectoryLogger(
        log_dir=".",
        log_action_csv_freq=1,
        log_state_csv_freq=1,
        log_state_render_freq=500
    )
    spinup_logger_kwargs = dict(output_dir=".", exp_name='log_files')
    env_builder = lambda: env_fn(trajactory_logger)
    vpg(env_fn=env_builder,
        actor_critic=cnn_actor_critic,
        ac_kwargs=AC_KWARGS,
        steps_per_epoch=N_STEPS_PER_EPOCH,
        epochs=EPOCHS,
        max_ep_len=N_STEPS_PER_EPISODE,
        logger_kwargs=spinup_logger_kwargs,
        save_freq=200,
        lam=0.97
    )


if __name__ == "__main__":
    main()

