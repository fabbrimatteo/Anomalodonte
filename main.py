# -*- coding: utf-8 -*-
# ---------------------

import os

import click
import torch.backends.cudnn as cudnn


# get project root
root = os.path.dirname(os.path.abspath(__file__))

# add project root to PYTHONPATH
python_path = os.environ.get('PYTHONPATH', None)
if python_path is None:
    python_path = root
else:
    python_path = f'{root}:{python_path}'
os.environ['PYTHONPATH'] = python_path

cudnn.benchmark = True


@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, conf_file_path, seed):
    # type: (str, str, int) -> None

    from conf import Conf
    from trainer import Trainer

    # if `exp_name` is None,
    # ask the user to enter it
    if exp_name is None:
        exp_name = click.prompt('▶ experiment name', default='default')

    # if `exp_name` contains '!',
    # `log_each_step` becomes `False`
    log_each_step = True
    if '!' in exp_name:
        exp_name = exp_name.replace('!', '')
        log_each_step = False

    # if `exp_name` contains a '@' character,
    # the number following '@' is considered as
    # the desired random seed for the experiment
    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name, log=log_each_step)
    print(f'\n{cnf}')

    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    trainer = Trainer(cnf=cnf)
    trainer.run()


if __name__ == '__main__':
    main()
