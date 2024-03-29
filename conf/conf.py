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


    @property
    def dict_view(self):
        x = self.__dict__
        y = {}
        for key in x:
            if not key in self.keys_to_hide:
                y[key] = x[key]
        return y


    def __init__(self, conf_file_path=None, seed=None, exp_name=None,
                 log=True, proj_log_path=None, yaml_file_path=None):
        # type: (str, int, str, bool) -> None
        """
        :param conf_file_path: optional path of the configuration file
        :param seed: desired seed for the RNG;
            >> if `None`, it will be chosen randomly
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
        if proj_log_path is None:
            self.proj_log_path = Path(__file__).parent.parent.abspath() / 'log'
        else:
            self.proj_log_path = Path(proj_log_path)

        self.exp_log_path = self.proj_log_path / self.exp_name  # type: Path

        # set random seed
        self.seed = set_seed(seed)  # type: int

        self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        if yaml_file_path is None:
            tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        else:
            tmp = yaml_file_path
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuration file
        if conf_file_path is None:
            y = {}
        else:
            conf_file = open(conf_file_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.Loader)

        # ---------------------------------------------------
        # read configuration parameters from YAML file
        # or set their default value

        # (1) training parameters
        self.lr = y.get('LR', 0.0001)  # type: float
        self.epochs = y.get('EPOCHS', 10)  # type: int
        self.max_patience = y.get('MAX_PATIENCE', 8)  # type: int
        self.pretrained_weights_path = \
            y.get('PRETRAINED_WEIGHTS_PATH', None)  # type: str

        # (2) dataset/dataloader parameters
        self.ds_path = y.get('DS_PATH', None)  # type: str
        self.master_ds_path = y.get('MASTER_DS_PATH', None)  # type: str
        self.batch_size = y.get('BATCH_SIZE', 8)  # type: int
        self.cam_id = y.get('CAM_ID', 'cam_1')  # type: str
        self.n_workers = y.get('N_WORKERS', 4)  # type: int
        self.data_aug = y.get('DATA_AUG', False)  # type: bool

        # (3) loss parameters
        self.loss_fn = y.get('LOSS_FN', 'L1+MS_SSIM')  # type: str
        self.rec_loss_w = y.get('REC_LOSS_W', 1)  # type: float
        self.int_loss_w = y.get('INT_LOSS_W', 100)  # type: float

        # (4) autoencoder bottleneck/code parameters
        self.code_channels = y.get('CODE_CHANNELS', 4)  # type: int
        self.code_h = y.get('CODE_H', None)  # type: Optional[int]
        self.code_w = y.get('CODE_W', None)  # type: Optional[int]
        # ---------------------------------------------------

        # select CPU or GPU
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = y.get('DEVICE', default_device)  # type: str

        # check if dataset exists
        assert self.ds_path is not None, \
            f'`you must set `DS_PATH` in configuration file`'
        assert self.master_ds_path is not None, \
            f'`you must set `MASTER_DS_PATH` in configuration file`'
        self.ds_path = Path(self.ds_path)
        self.master_ds_path = Path(self.master_ds_path)
        assert self.ds_path.exists(), \
            f'`DS_PATH: "{self.ds_path.abspath()}"` does not exists'
        assert self.master_ds_path.exists(), \
            f'`DS_PATH: "{self.master_ds_path.abspath()}"` does not exists'


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
    cnf = Conf(exp_name='cam1')
    cnf.write_to_file('./default.yaml')
    print(f'\nDefault configuration parameters: \n{cnf}')


if __name__ == '__main__':
    show_default_params()
