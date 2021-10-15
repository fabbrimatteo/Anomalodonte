# -*- coding: utf-8 -*-
# ---------------------

import os


PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional
import termcolor


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()

    if HOSTNAME == 'shenron':
        NAS_PATH = Path('/nas/softechict-nas-3/matteo')
    elif HOSTNAME == 'sarah-kennedy':
        NAS_PATH = Path('/run/user/1000/gvfs/sftp:host=shenron.ing.unimo.it,user=matteo')
        NAS_PATH = NAS_PATH / 'nas/softechict-nas-3/matteo'
    else:
        NAS_PATH = Path('/nas/softechict-nas-3/matteo')


    @property
    def dict_view(self):
        x = self.__dict__
        y = {}
        for key in x:
            if not key in self.keys_to_hide:
                y[key] = x[key]
        return y


    def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True):
        # type: (str, int, str, bool) -> None
        """
        :param conf_file_path: optional path of the configuration file
        :param seed: desired seed for the RNG; if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param log: `True` if you want to log each step; `False` otherwise
        """
        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        self.project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_log_path = Path(__file__).parent.parent.abspath() / 'log'
        # alternative: self.project_log_path = Path(Conf.NAS_PATH / 'log' / self.project_name)
        self.exp_log_path = self.project_log_path / exp_name  # type: Path

        # set random seed
        self.seed = set_seed(seed)  # type: int

        self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuation file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.Loader)

        # read configuration parameters from YAML file
        # or set their default value
        self.lr = y.get('LR', 0.0001)  # type: float
        self.epochs = y.get('EPOCHS', 10)  # type: int
        self.n_workers = y.get('N_WORKERS', 4)  # type: int
        self.batch_size = y.get('BATCH_SIZE', 8)  # type: int
        self.loss_fn = y.get('LOSS_FN', 'L1+MS_SSIM+VGG')  # type: str
        self.rec_error_fn = y.get('REC_ERROR_FN', 'CODE_MSE_LOSS')  # type: str
        self.max_patience = y.get('MAX_PATIENCE', 8)  # type: int
        self.code_channels = y.get('CODE_CHANNELS', 4)  # type: int
        self.code_h = y.get('CODE_H', None)  # type: Optional[int]
        self.code_w = y.get('CODE_W', None)  # type: Optional[int]
        self.code_noise = y.get('CODE_NOISE', 0.25)  # type: float
        self.ds_path = y.get('DS_PATH', None)  # type: str

        # pre processing params
        self.resized_h = y.get('RESIZED_H', 256)  # type: int
        self.resized_w = y.get('RESIZED_W', 256)  # type: int
        self.crop_x_min = y.get('CROP_X_MIN', 812)  # type: int
        self.crop_y_min = y.get('CROP_Y_MIN', 660)  # type: int
        self.crop_side = y.get('CROP_SIDE', 315)  # type: int

        self.ds_path = Path(self.ds_path)

        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = y.get('DEVICE', default_device)  # type: str


    def write_to_file(self, out_file_path):
        # type: (str) -> None
        """
        Writes configuration parameters to `out_file_path`
        :param out_file_path: path of the output file
        """
        import re

        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape.sub('', str(self))
        with open(out_file_path, 'w') as out_file:
            print(text, file=out_file)


    def __str__(self):
        # type: () -> str
        out_str = ''
        dw = self.dict_view
        for key in dw:
            value = dw[key]
            if type(value) is Path or type(value) is str:
                value = value.replace(Conf.NAS_PATH, '$CAPRA_PATH')
                value = termcolor.colored(value, 'yellow')
            else:
                value = termcolor.colored(f'{value}', 'magenta')
            out_str += termcolor.colored(f'{key.upper()}', 'blue')
            out_str += termcolor.colored(': ', 'red')
            out_str += value
            out_str += '\n'
        return out_str[:-1]


    def no_color_str(self):
        # type: () -> str
        out_str = ''
        dw = self.dict_view
        for key in dw:
            value = dw[key]
            if type(value) is Path or type(value) is str:
                value = value.replace(Conf.NAS_PATH, '$NAS_PATH')
            out_str += f'{key.upper()}: {value}\n'
        return out_str[:-1]


def show_default_params():
    """
    Print default configuration parameters
    """
    cnf = Conf(exp_name='default')
    print(f'\nDefault configuration parameters: \n{cnf}')


if __name__ == '__main__':
    show_default_params()
