import argparse
import models
import sys
import datetime
import time
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default=None,
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='errnet_model', help='chooses which model to use.',
                                 choices=model_names)
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_epoch', '-re', type=int, default=None,
                                 help='checkpoint to use. (default: latest')
        self.parser.add_argument('--seed', type=int, default=2018, help='random seed to use. Default=2018')

        # for setting input
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=None,
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for display
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--no_verbose', action='store_true', help='disable verbose info?')
        self.parser.add_argument('--no_log', action='store_true', help='disable tf logger?')
        self.parser.add_argument('--not_host', action='store_true', help='remote or host?')
        self.initialized = True

def get_command_run():
    args = sys.argv.copy()
    args[0] = args[0].split('/')[-1]

    if sys.version[0] == '3':
        command = 'python3'
    else:
        command = 'python'

    for i in args:
        command += ' ' + i
    return command

def get_time_stamp(add_offset=0):
    """Get time_zone+0 unix time stamp (seconds)

    Args:
        add_offset(int): bias added to time stamp

    Returns:
        (str): time stamp seconds
    """
    ti = int(time.time())
    ti = ti + add_offset
    return str(ti)


def get_time_str(time_stamp=get_time_stamp(), fmt="%Y/%m/%d %H:%M:%S", timezone=8, year_length=4):
    """Get formatted time string.

    Args:
        time_stamp(str): linux time string (seconds).
        fmt(str): string format.
        timezone(int): time zone.
        year_length(int): 2 or 4.

    Returns:
        (str): formatted time string.

    Example:
        >>> get_time_str()
        >>> # 2020/01/01 13:30:00

    """
    if not time_stamp:
        return ''

    time_stamp = int(time_stamp)

    base_time = datetime.datetime.utcfromtimestamp(time_stamp)

    time_zone_time = base_time + datetime.timedelta(hours=timezone)
    format_time_str = time_zone_time.strftime(fmt)

    if year_length == 2:
        format_time_str = format_time_str[2:]
    return format_time_str

with open('run_log.txt', 'a') as f:
    f.writelines(get_time_str(fmt="%Y-%m-%d %H:%M:%S") + ' ' + get_command_run() + '\n')