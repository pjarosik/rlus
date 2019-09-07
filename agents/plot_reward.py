import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

AGGR = {
    "mean": np.mean,
    "sum": np.sum
}


def _get_cumulative_reward(ep_dir, aggr="sum"):
    with open(os.path.join(ep_dir, 'action.csv')) as f:
        reader = csv.DictReader(f, delimiter='\t')
        rewards = np.array([float(row['reward']) for row in reader])
        return AGGR[aggr](rewards)


def _get_cumulative_l1_distance(ep_dir, aggr="sum"):
    with open(os.path.join(ep_dir, 'state.csv')) as f:
        reader = csv.DictReader(f, delimiter='\t')
        poss = np.array([
            (float(row['probe_x']), float(row['probe_z']), float(row['obj_x']), float(row['obj_z']))
            for row in reader
        ])
    probe_pos = poss[:, :2]
    obj_pos = poss[:, 2:]
    ret = np.abs(probe_pos-obj_pos)
    ret = AGGR[aggr](ret)
    return ret


def _get_cumulative_angle_diff(ep_dir, aggr="sum"):
    with open(os.path.join(ep_dir, 'state.csv')) as f:
        reader = csv.DictReader(f, delimiter='\t')
        angles = np.array([(float(row['probe_angle']), float(row['obj_angle'])) for row in reader])
    ret = np.abs((angles[:, 0]%360)-(angles[:, 1]%360))
    ret = AGGR[aggr](ret)
    return ret


def _compute_mean_std(values, window_size):
    mean, std = [], []
    if window_size % 2:
        rn = range(-(window_size//2), len(values)-(window_size//2))
    else:
        rn = range(-(window_size//2-1), len(values)-(window_size//2-1))
    for i in rn:
        l = 0 if i < 0 else i
        r = i+window_size
        mean.append(np.mean(values[l:r]))
        std.append(np.std(values[l:r]))
    x = np.arange(0, len(values))
    mean, std = np.array(mean), np.array(std)
    assert len(x) == len(mean) == len(std)
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
        max_ep=None,
        dec=50,
        aggr="sum"
):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)

    c_rewards = []
    c_ref_metric = []
    c_angle_metric = []

    episode_nr = 0
    ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)

    while os.path.exists(os.path.join(ep_dir, "action.csv")) and \
          os.path.exists(os.path.join(ep_dir, "state.csv")) and \
          (max_ep is None or episode_nr <= max_ep):
        c_rewards.append(_get_cumulative_reward(ep_dir, aggr=aggr))
        c_ref_metric.append(_get_cumulative_l1_distance(ep_dir, aggr=aggr))
        c_angle_metric.append(_get_cumulative_angle_diff(ep_dir, aggr=aggr))
        episode_nr += 1
        ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)

    def _plot_values(x, mean, std, dec):
        _, caplines, _ = ax.errorbar(
            x, mean, yerr=std, errorevery=dec,
            uplims=True, lolims=True)
        for c in caplines:
            c.set_marker("_")
        ax.grid(linestyle="--")
        # caplines[0].set_markersize(20)
        # ax.plot(x, mean)
        # ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    if type == "reward":
        x, c_rewards_mean, c_rewards_std = _compute_mean_std(c_rewards, window_size)
        _plot_values(x, c_rewards_mean, c_rewards_std, dec=dec)
        ax.set_ylabel('Reward  $R(\\tau)$')
    elif type == "l1":
        x, c_ref_metric_mean, c_ref_metric_std = _compute_mean_std(c_ref_metric, window_size)
        _plot_values(x, c_ref_metric_mean, c_ref_metric_std, dec=dec)
        ax.set_ylabel("Cumulative $L_1$")
    elif type == "angle":
        x, c_angle_mean, c_angle_std = _compute_mean_std(c_angle_metric, window_size)
        _plot_values(x, c_angle_mean, c_angle_std, dec=dec)
        ax.set_ylabel('Cumulative $\Delta \\alpha$')
    ax.set_xlabel("Episode")
    fig.savefig(os.path.join(exp_dir, "%s.pdf" % type), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot reward for each epoch/episode.")
    parser.add_argument("--exp_dir", dest="exp_dir",
                        help="Path to the directory with experiment results.",
                        required=True)
    parser.add_argument("--type", dest="type",
                        help="Type of the plot to produce.",
                        choices=["reward", "l1", "angle"],
                        required=True)
    parser.add_argument("--aggr", dest="aggr",
                        help="Episode return aggregation type.",
                        choices=["sum", "mean"],
                        required=False,
                        default="sum")
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
                max_ep=args.max_ep, aggr=args.aggr)








