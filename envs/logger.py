import os
import csv
from PIL import Image


class TrajectoryLogger:
    def __init__(self,
                 log_dir=None,
                 log_action_csv_freq=1,
                 log_state_csv_freq=1,
                 log_state_render_freq=10,  # 0 or None means do not log info
                 state_views=('env', 'observation')
                 ):

        """
        Args:
            log_dir: where to store logging information
            log_action_csv_freq: how often performed actions should be logged,
                                 for example, when freq=10, actions will be logged
                                 after every ten episodes; when False,
                                 do not log actions to CSV files
            log_state_csv_freq:  how often obtained states should be logged,
                                 for example, when freq=10, env states will be logged
                                 after every ten episodes; when False,
                                 do not log the data
            log_state_render_freq: how often env.renders should be logged to file,
                                  for example, when freq=10, env will be rendered to files
                                  after every ten episodes; when False,
                                  do not log the data
        """
        self.log_action_csv_freq = log_action_csv_freq
        self.log_state_csv_freq = log_state_csv_freq
        self.log_state_render_freq = log_state_render_freq
        self.state_views = state_views
        self.log_dir = log_dir
        self.action_logger, self.state_logger = None, None
        self.state_render_loggers = []

    def restart(self, env, episode_nr):
        """
        Restarts trajectory recorder, creates all necessary resources
        required to save logs (directory tree structure, etc.).

        Should be called at the start of the new episode.

        Args:
            episode_nr: the number of the next episode.
        """
        episode_dir = os.path.join(self.log_dir, "episode_%d" % episode_nr)
        os.makedirs(episode_dir, exist_ok=True)
        if self.log_action_csv_freq:
            log_file = os.path.join(episode_dir, "action.csv")
            self.action_logger = CSVLogger(
                log_file, fieldnames=[
                    "step",
                    "action_code",
                    "action_name",
                    "reward"
                ])
        if self.log_state_csv_freq:
            log_file = os.path.join(episode_dir, "state.csv")
            self.state_logger = CSVLogger(
                log_file, fieldnames=[
                    "step",
                    "probe_x",
                    "probe_z",
                    "probe_angle",
                    "obj_x",
                    "obj_z",
                    "obj_angle",
                ])
        if self.log_state_render_freq and self.state_views:
            self.state_render_loggers = []
            for v in self.state_views:
                self.state_render_loggers.append(UsPhantomEnvRenderLogger(episode_dir, v))

    def log_action(self, episode, step, action_code, reward, action_name=None):
        if episode % self.log_action_csv_freq == 0:
            self.action_logger.log(
                step=step,
                action_code=action_code,
                action_name=action_name,
                reward=reward
            )

    def log_state(self, episode, step, env):
        if episode % self.log_state_csv_freq == 0:
            state = env.get_state_desc()
            state['step'] = step
            self.state_logger.log(**state)
        if episode % self.log_state_render_freq == 0:
            for logger in self.state_render_loggers:
                logger.log(step, env)


class CSVLogger:
    def __init__(self, output_file, fieldnames):
        self.fieldnames = fieldnames
        self.output_file = output_file
        self._write_header = True

    def log(self, **kwargs):
        with open(self.output_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames,delimiter='\t')
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow(kwargs)


class UsPhantomEnvRenderLogger:
    def __init__(self, output_dir, view):
        self.output_dir = output_dir
        self.view = view

    def log(self, step, env):
        r = env.render(mode='rgb_array', views=[self.view])
        img = Image.fromarray(r, mode="RGB")
        img.save(os.path.join(self.output_dir, "%s_step_%03d.png" % (self.view, step)),
                 dpi=(300,300), optimize=False, compress_level=1)

