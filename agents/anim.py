import argparse
import os
import csv
from PIL import Image, ImageDraw, ImageFont


def _merge(img1, img2):
    widths, heights = zip(*(img1.size, img2.size))
    width, height = sum(widths), max(heights)
    ret = Image.new("RGB", (width, height))
    ret.paste(img1, (0, 0))
    ret.paste(img2, (img1.size[0], 0))
    return ret


def _get_rewards(ep_dir):
    with open(os.path.join(ep_dir, 'action.csv')) as f:
        reader = csv.DictReader(f, delimiter='\t')
        return [float(row['reward']) for row in reader]


def anim(exp_dir,
         min_ep=0,
         max_ep=None,
         step=1,
         interval=100,
         font=None,
         font_size=None,
         width=None,
         height=None,
         ep_pause=None):
    episode_nr = min_ep
    internal_ep_nr = 0
    frames = []
    ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)
    font = ImageFont.truetype(font=font, size=font_size) if font else None
    while os.path.exists(os.path.join(ep_dir, "action.csv")) and \
          os.path.exists(os.path.join(ep_dir, "state.csv")) and \
          (max_ep is None or episode_nr <= max_ep):

        rewards = _get_rewards(ep_dir)
        if internal_ep_nr % step == 0:
            env_path_pattern = os.path.join(ep_dir, "env_step_%03d.png")
            obs_path_pattern = os.path.join(ep_dir, "observation_step_%03d.png")
            s = 0
            img = None
            while os.path.exists(env_path_pattern % s) and \
                  os.path.exists(obs_path_pattern % s):

                env = Image.open(env_path_pattern % s)
                obs = Image.open(obs_path_pattern % s)
                if width and height:
                    env = env.resize((width, height), Image.ANTIALIAS)
                    obs = obs.resize((width, height), Image.ANTIALIAS)
                img = _merge(env, obs)
                draw = ImageDraw.Draw(img)
                text = "Episode: %04d Step: %02d" % (episode_nr, s)
                if s > 0:
                    text = text + " Reward %.02f" % rewards[s-1]
                draw.text((10, 10), text, (0, 0, 0), font=font)
                frames.append(img)
                s += 1
            if ep_pause:
                frames.extend([img]*(ep_pause-1))
            internal_ep_nr += 1
        episode_nr += 1
        ep_dir = os.path.join(exp_dir, "episode_%d" % episode_nr)

    output_path = os.path.join(exp_dir, "render.gif")
    frames[0].save(output_path, format='GIF', append_images=frames[1:],
                   save_all=True, duration=interval, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate given range of episodes.")
    parser.add_argument("--exp_dir", dest="exp_dir",
                        help="Path to the directory with experiment results.",
                        required=True)
    parser.add_argument("--min_ep", dest="min_ep",
                        help="Number of the first episode to include in "
                             "animation.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--step", dest="step",
                        help="How many available episodes to skip between "
                             "consecutive two animations.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--max_ep", dest="max_ep",
                        help="Maximum number of episodes to consider.",
                        type=int,
                        default=None,
                        required=False)
    parser.add_argument("--interval", dest="interval",
                        help="Time interval length between consecutive frames.",
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument("--font", dest="font",
                        help="Path to the ttf font file, that should be used "
                             "for captions.",
                        default=None,
                        required=False)
    parser.add_argument("--font_size", dest="font_size",
                        help="The size of font, which will  be used "
                             "for captions.",
                        type=int,
                        default=None,
                        required=False)
    parser.add_argument("--width", dest="width",
                        help="Width of the image with state/observation ("
                             "output GIF will have width = 2*state_width). ",
                        type=int,
                        default=None,
                        required=False)
    parser.add_argument("--height", dest="height",
                        help="Height of the image with state/observation.",
                        type=int,
                        default=None,
                        required=False)
    parser.add_argument("--ep_pause", dest="ep_pause",
                        help="How many times the last step of each episode "
                             "should be displayed (adds short pause at the "
                             "end of the episode).",
                        type=int,
                        default=None,
                        required=False)

    args = parser.parse_args()
    anim(
        args.exp_dir, min_ep=args.min_ep, max_ep=args.max_ep, step=args.step,
        interval=args.interval, font=args.font, font_size=args.font_size,
        width=args.width, height=args.height, ep_pause=args.ep_pause)







