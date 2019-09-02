from envs.us_env import PhantomUsEnv, random_env_generator
import envs.us_env as phantom_env
import argparse
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Generates given number of cyst phantom RF examples.""")

    parser.add_argument("--episodes", dest="no_episodes", type=int,
                        help="Number of episodes to generate.",
                        required=True)
    parser.add_argument("--steps", dest="no_steps", type=int,
                        help="Max episode length.",
                        required=True)
    parser.add_argument("--workers", dest="no_workers", type=int,
                        help="Number of MATLAB workers.",
                        required=True)
    parser.add_argument("--output_dir", dest="output_dir",
                        help="Destination directory for observations and env. vis.",
                        required=True)

    args = parser.parse_args()

    imaging = phantom_env.ImagingSystem(
        c=1540,
        fs=100e6,
        image_width=40/1000,
        image_resolution=(40, 90),
        # image_grid=(40/1000, 90/1000),
        grid_step=0.5/1000,
        no_lines=64,
        median_filter_size=5,
        dr_threshold=-100
    )
    env = PhantomUsEnv(
        imaging=imaging,
        env_generator=random_env_generator(),
        max_steps=args.no_steps,
        no_workers=args.no_workers
    )

    def plot_obj(d, i, ob, title):
        fig = plt.figure()
        plt.title(title)
        plt.imshow(ob, cmap='gray')
        plt.savefig(os.path.join(d, "step_%03d.png" % i))

    def plot_env(d, i):
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d')
        env.render_pyplot(ax)
        plt.savefig(os.path.join(d, "env_%03d.png" % i))


    for episode in range(args.no_episodes):
        print("Episode %d" % episode)
        env.reset()
        step = 0
        episode_dir = os.path.join(args.output_dir, "episode_%d" % episode)
        # just to obtain initial observation
        ob, reward, episode_over, _ = env.step([0,0,0])
        if args.output_dir:
            os.makedirs(episode_dir, exist_ok=True)
            plot_env(episode_dir, step)
            plot_obj(episode_dir, step, ob, "ep: %d, step: %d, reward: %s" % (episode, step, str(reward)))
        while not episode_over:
            step += 1
            print("Step %d" % step)
            action = env.action_space.sample()
            print("Performing action: %s" % str(action))
            start = time.time()
            ob, reward, episode_over, _ = env.step(action)
            end = time.time()
            print("Environment execution time: %d [s]" % (end-start))
            print("reward %f" % reward)
            with open(os.path.join(episode_dir, "log.txt"), 'a') as f:
                f.write("Episode %d, step %d\n" % (episode, step))
                f.write("Take action: %s\n" % action)
                f.write("Reward: %f\n" % reward)
                f.write(env.to_string())
            print(env.to_string())
            if args.output_dir:
                plot_obj(episode_dir, step, ob, "ep: %d, step: %d, reward: %s" % (episode, step, str(reward)))
                plot_env(episode_dir, step)
    print("Training is over (achieved max. number of episodes).")


