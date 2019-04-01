from spinup import ppo, vpg
import spinup.algos.vpg.core as core
import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Box, Discrete
import envs.phantom_env as phantom_env
from envs.phantom_env import UsPhantomEnv, random_env_generator, const_env_generator


# below code bases on openai.spinup A-C scheme implementation
def cnn(x,
        hidden_sizes=(32,),
        kernel_size=(3,3),
        pool_size=(2,2),
        activation=tf.tanh,
        output_activation=None
):
    for h in hidden_sizes[:-1]:
        x = tf.layers.conv2d(x, filters=h, kernel_size=kernel_size, activation=activation)
        # TODO batch normalization
        x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=pool_size)
    x = tf.layers.flatten(x)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def cnn_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = cnn(x, hidden_sizes=list(hidden_sizes)+[act_dim], activation=activation, output_activation=None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1) # action drawn from current policy
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1) # log probability of given actions
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1) # log probability of actions of given pi
    return pi, logp, logp_pi


def cnn_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = cnn(x, hidden_sizes=list(hidden_sizes)+[act_dim], activation=activation, output_activation=output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = core.gaussian_likelihood(a, mu, log_std)
    logp_pi = core.gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


def cnn_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = cnn_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = cnn_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(cnn(x, hidden_sizes=list(hidden_sizes)+[1], activation=activation, output_activation=None), axis=1)
    return pi, logp, logp_pi, v


N_STEPS_PER_EPISODE = 16
EPOCHS = 10
N_WORKERS = 4
EXP_DIR = '/home/pjarosik/src/rlus/agents/us_phantom'

def env_fn():
    probe = phantom_env.Probe(
        pos=np.array([0, 0, 0]),
        angle=0,
        width=40/1000,
        height=10/1000,
        focal_depth=60/1000
    )
    phantom = phantom_env.Phantom(
        objects=[
            phantom_env.Teddy(
                belly_pos=np.array([15/1000, 0, 40/1000]),
                scale=10/1000,
                dist_ahead=.9
            )
            .rotate_xy(angle=60),
        ],
        x_border=(-60/1000, 60/1000),
        y_border=(-60/1000, 60/1000),
        z_border=(0, 90/1000),
        n_scatterers=int(4e4),
        n_bck_scatterers=int(2e3)
    )
    imaging = phantom_env.Imaging(
        c=1540,
        fs=100e6,
        image_width=40/1000,
        image_height=90/1000,
        image_resolution=(40, 90), # [pixels]
        no_lines=64,
        median_filter_size=5,
        dr_threshold=-100,
        dec=2
    )
    env = UsPhantomEnv(
        imaging=imaging,
        env_generator=const_env_generator(phantom, probe),
        max_steps=N_STEPS_PER_EPISODE,
        no_workers=N_WORKERS,
        log_freq=2,
        log_dir=EXP_DIR
    )
    return env


ac_kwargs = dict(
    hidden_sizes=[8, 16],
    activation=tf.nn.relu
    # how about output_activation?
)

logger_kwargs = dict(output_dir=EXP_DIR, exp_name='initial_test')

vpg(
    env_fn=env_fn,
    actor_critic=cnn_actor_critic,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=N_STEPS_PER_EPISODE,
    epochs=EPOCHS,
    max_ep_len=N_STEPS_PER_EPISODE,
    logger_kwargs=logger_kwargs)
