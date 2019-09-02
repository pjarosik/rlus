import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np



def _get_cumulative_reward(ep_dir):
    with open(os.path.join(ep_dir, 'action.csv')) as f:
        reader = csv.DictReader(f, delimiter='\t')
        rewards = [float(row['reward']) for row in reader]
        return sum(rewards)


def _get_cumulative_l1_distance(ep_dir):
    with open(os.path.join(ep_dir, 'state.csv')) as f:
        reader = csv.DictReader(f, delimiter='\t')
        probe_pos = np.array([(float(row['probe_x']), float(row['probe_z'])) for row in reader])
    with open(os.path.join(ep_dir, 'state.csv')) as f:
        reader = csv.DictReader(f, delimiter='\t')
        obj_pos = np.array([(float(row['obj_x']), float(row['obj_z'])) for row in reader])
    ret = np.abs(probe_pos-obj_pos)
    ret = np.sum(ret)
    return ret


def _compute_mean_std(values, window_size):
    mean, std = [], []
    for i in range(len(values)):
        mean.append(np.mean(values[i:i + window_size]))
        std.append(np.std(values[i:i + window_size]))
    x = np.linspace(
        start=window_size/2, stop=len(mean)-window_size/2,
        num=len(mean))
    x =  np.concatenate(([0], x, [len(values)]))
    mean = np.concatenate(([values[0]], mean, [values[-1]]))
    std = np.concatenate(([0], std, [0]))
    # print("Mean")
    # print(mean.tolist())
    # print("Std")
    # print(std.tolist())
    return x, mean, std


def plot_reward(
        exp_dir,
        figsize,
        window_size=40,
        type='reward',
        max_ep=None
                ):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)

    c_rewards = []
    c_ref_metric = []

    episode_nr = 0
    ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)

    while os.path.exists(os.path.join(ep_dir, "action.csv")) and \
          os.path.exists(os.path.join(ep_dir, "state.csv")) and \
          (max_ep is None or episode_nr <= max_ep):
        c_rewards.append(_get_cumulative_reward(ep_dir))
        c_ref_metric.append(_get_cumulative_l1_distance(ep_dir))
        episode_nr += 1
        ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)

    def _plot_values(x, mean, std):
        ax.plot(x, mean)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    if type == "reward":
        x, c_rewards_mean, c_rewards_std = _compute_mean_std(c_rewards, window_size)
        _plot_values(x, c_rewards_mean, c_rewards_std)
        ax.set_ylabel('Reward  $R(\\tau)$')
    elif type == "l1":
        x, c_ref_metric_mean, c_ref_metric_std = _compute_mean_std(c_ref_metric, window_size)
        _plot_values(x, c_ref_metric_mean, c_ref_metric_std)
        ax.set_ylabel("Cumulative $L_1$")
    ax.set_xlabel("Episodes")
    fig.savefig(os.path.join(exp_dir, "%s.pdf" % type), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot reward for each epoch/episode.")
    parser.add_argument("--exp_dir", dest="exp_dir",
                        help="Path to the directory with experiment results.",
                        required=True)
    parser.add_argument("--type", dest="type",
                        help="Type of the plot to produce.",
                        choices=["reward", "l1"],
                        required=True)
    parser.add_argument("--max_ep", dest="max_ep",
                        help="Maximum number of episodes to consider.",
                        type=int,
                        default=None,
                        required=False)


    args = parser.parse_args()

    figsize = (10, 5)
    window_size = 40
    plot_reward(args.exp_dir, figsize=figsize,
                window_size=window_size, type=args.type,
                max_ep=args.max_ep)








