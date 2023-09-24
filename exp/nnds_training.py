#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np

from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.pardir, 'src'))

from learn_nn_ds import NL_DS
from utils.utils import mse, time_stamp
from utils.log_config import logger
from utils.data_loader import load_snake_data
from utils.plot_tools import plot_ds_stream, plot_trajectory, plot_contours
from learn_gmm_ds import prepare_expert_data


def train_neural_policy(network: str, mode: str = 'train', motion_shape: str = "G",
    n_dems: int = 10, n_epochs: int = 1000, plot: bool = False, model_name: str = 'test',
    test_size: float = 0.01, save: bool = True, save_dir: str = "."):
    """ Training sequence for a stable/unstable neural policy to estimate a
    nonlinear dynamical system.

    Args:
        network(str): Type of the nonlinear estimator, could be nn, snds, sdsef.
        motion_shape (str, optional): Shape of the trajectories. Defaults to "G".
        n_dems (int, optional): Number of augmented demonstrations. Defaults to 10.
        plot (bool, optional): Whether to plot trajectories and final ds or not. Defaults to False.
        n_epochs (int, optional): Total number of epochs. Defaults to 1000.
        model_name (str, optional): Name of the model for save and load. Defaults to 'test'.
        test_size (float, optional): Size of the test dataset. Defaults to 0.2.
        save_dir (str, optional): In case save is activated, files will be saved in this directory.
    """

    ''' Load an augmented dataset '''
    model_name = model_name.lower()
    name = f'{model_name}-{network}-{motion_shape.lower()}-{time_stamp()}'

    aug_trajs, aug_vels = prepare_expert_data(motion_shape, n_dems,
        dir=os.path.join(os.pardir, 'res', 'expert_models'))

    trajs_train, trajs_test, vels_train, vels_test = train_test_split(aug_trajs, aug_vels,
        test_size=test_size, random_state=np.random.randint(10))
    logger.info(f'Shape of the train data is {trajs_train.shape} and test is {trajs_test.shape}.')

    plot_trajectory(aug_trajs)

    ''' Train and save a model'''
    nl_ds = NL_DS(network=network, data_dim=aug_trajs.shape[1])

    if mode == 'train':
        nl_ds.fit(trajs_train, vels_train, n_epochs=n_epochs, trajectory_test=trajs_test, velocity_test=vels_test)

    if mode == 'test':
        nl_ds.load(model_name, dir=save_dir)

    ''' Plot the DS '''
    if plot:
        plot_ds_stream(nl_ds, aug_trajs[len(aug_trajs) - 7000:], save_dir=save_dir, file_name=name,
                       show_legends=True, space_stretch=0.5)

        if nl_ds.lpf() is not None:
            plot_contours(nl_ds.lpf, aug_trajs, save_dir=save_dir, file_name=f'{name}-lpf')

    ''' Save the DS '''
    if save:
        nl_ds.save(model_name=name, dir=save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')

    parser.add_argument('-nt', '--neural-tool', type=str, default="nn",
        help='The neural policy or tool among snds, nn, sdsef.')
    parser.add_argument('-m', '--mode', type=str, default="train",
        help='Mode between train and test. Test mode only loads the model with the provided name.')
    parser.add_argument('-ms', '--motion-shape', type=str, default="G",
        help='Shape of the trajectories as in LASA dataset.')
    parser.add_argument('-nd', '--num-demonstrations', type=int, default=7,
        help='Number of additional demonstrations to the original dataset.')
    parser.add_argument('-ne', '--num-epochs', type=int, default=10000,
        help='Number of training epochs.')
    parser.add_argument('-sp', '--show-plots', action='store_true', default=False,
        help='Show extra plots of final result and trajectories.')
    parser.add_argument('-sm', '--save-model', action='store_true', default=False, help='Save the model in the res folder.')
    parser.add_argument('-sd', '--save-dir', type=str,
        default=os.path.join(os.pardir, 'res', 'nlds_policy'),
        help='Optional destination for save/load.')
    args = parser.parse_args()

    train_neural_policy(args.nonlinear_tool, args.mode, args.motion_shape,
        args.num_demonstrations, args.num_epochs, args.show_plots,
        save=args.save_model, save_dir=args.save_dir)
