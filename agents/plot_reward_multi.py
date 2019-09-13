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



def plot_reward(
        exp_dirs,
        figsize,
        type='reward',
        max_ep=None,
        dec=50,
        aggr="sum"
):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)

    data = []

    extr = {
        "reward": lambda ep_dir: _get_cumulative_reward(ep_dir, aggr=aggr),
        "l1": lambda ep_dir: _get_cumulative_l1_distance(ep_dir, aggr=aggr),
        "angle": lambda ep_dir: _get_cumulative_angle_diff(ep_dir, aggr=aggr),
    }[type]
    y_label = {
        "reward": 'Reward  $R(\\tau)$',
        "l1":     "%s $L_1$",
        "angle":  "%s (\\alpha-\\alpha_o)"
    }[type]

    for exp_dir in exp_dirs:
        ep_data = []
        episode_nr = 0
        ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)

        while os.path.exists(os.path.join(ep_dir, "action.csv")) and \
          os.path.exists(os.path.join(ep_dir, "state.csv")) and \
          (max_ep is None or episode_nr <= max_ep):
            ep_data.append(extr(ep_dir))
            episode_nr += 1
            ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)
        data.append(ep_data)

    def _plot_values(x, mean, std, dec):
        _, caplines, _ = ax.errorbar(
            x, mean, yerr=std, errorevery=dec,
            uplims=True, lolims=True)
        for c in caplines:
            c.set_marker("_")
        ax.grid(linestyle="--")

    data = np.array(data)
    print(data.shape)

    mean = np.mean(data, axis=0)
    std = np.mean(data, axis=0)
    x = np.arange(len(mean))
    _plot_values(x, mean, std, dec=dec)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    fig.savefig("%s_%s.pdf" % (exp_dirs[0], type), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot reward for each epoch/episode.")
    parser.add_argument("--exp_dirs", dest="exp_dirs",
                        help="Path to the directories with experiments.",
                        required=True,
                        nargs="+"
                        )
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
    print(args.exp_dirs)

    figsize = (10, 5)
    plot_reward(
        args.exp_dirs,
        figsize=figsize,
        type=args.type,
        max_ep=args.max_ep,
        aggr=args.aggr)








