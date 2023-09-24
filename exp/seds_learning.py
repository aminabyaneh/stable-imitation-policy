#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

from typing import Tuple

sys.path.append(os.path.join(os.pardir, 'src'))

from learn_gmm_ds import SE_DS
from utils.utils import mse, time_stamp
from utils.log_config import logger
from policy_interface import PlanningPolicyInterface
from utils.plot_tools import plot_ds_stream, plot_trajectory, plot_ds_stream
from utils.data_loader import generate_synthetic_data_trajs, load_pylasa_data


def learn_seds_synthetic_data(show_plots: bool = False):
    """ Testing whether synthetic data generation tools and loading tools
    work properly.

    Args:
        show_plots (bool, optional): Whether to show plots at each step.
            Defaults to False.
    """

    ''' Synthetic data generation example '''
    ds_a_mat = np.matrix([[-1., 0.], [0., -1.]], dtype=np.float64)
    ds_b_mat = np.array([0., 0.], dtype=np.float64)

    positions_syn, velocities_syn = generate_synthetic_data_trajs(A=ds_a_mat,
                                        start_point=(4., -4.), goal_point=(0., 0.),
                                        n_samples=7, start_dev=0.2, traj_dev=0.1)
    if show_plots:
        plot_trajectory(positions_syn, title=f'Demonstration for Synthetic Data')

    ''' Demo on synthetic linear data'''
    demo = SE_DS()
    demo.fit(positions_syn, velocities_syn)

    ''' Plot results '''
    if show_plots:
        plot_ds_stream(demo, positions_syn, "Synthetic data DS", frequency=50,
                       scale_factor=10, width=0.05)


def learn_seds_policy(motion_shape: str, show_plots: bool = False,
        model_name: str = "test", n_dems: int = 7, save: bool = False,
        save_dir : str = os.path.join(os.pardir, 'res', 'seds_policy')) -> Tuple[float, PlanningPolicyInterface]:
    """ Learning a demonstrated motion and showing the results.

    Args:
        motion_shape (str): Shape of the demonstrated motion in handwriting dataset.
        show_plots (bool, optional): Whether to show plots at each step.
            Defaults to False.
        save (bool, optional): Whether to save the seds model or not. Default is False.

    Returns:
        Tuple[float, PlanningPolicyInterface]: MSE and the model itself
    """
    name = f'{model_name.lower()}-{motion_shape.lower()}-{time_stamp()}'

    ''' Handwriting dataset '''
    positions_py, velocities_py = load_pylasa_data(motion_shape=motion_shape, n_dems=n_dems)
    logger.info(f'Handwriting dataset loaded with [{positions_py.shape}, {velocities_py.shape}] samples.')

    ''' Learn on demonstration data '''
    lpvds = SE_DS()
    lpvds.fit(positions_py, velocities_py)

    ''' Plot results '''
    if show_plots:
        plot_ds_stream(lpvds, positions_py, save_dir=save_dir, file_name=f'ds-{name}',
                       show_legends=False)

    ''' Save the model '''
    if save:
        lpvds.save(model_name=name, dir=save_dir)

    ''' Evaluate both models on prediction '''
    positions_py, velocities_py = load_pylasa_data(motion_shape=motion_shape, normalized=True)
    pred = lpvds.predict(positions_py)
    error = mse(pred, velocities_py)
    logger.info(f'MSE utilizing the entire dataset is {error}.')

    return error, lpvds


def main():
    """ Main entry point and argument parser for the exp file.
    """

    parser = argparse.ArgumentParser(description='Handle basic experiments for learning DS.')
    parser.add_argument('-syd', '--synthetic-demo', action='store_true', default=False,
                        help='Run SEDS with synthetic data.')
    parser.add_argument('-pd', '--plot-demo', type=str, default="",
                        help='Pass the model name to plot an SEDS policy.')
    parser.add_argument('-sp', '--show-plots', action='store_true', default=False,
                        help='Set false to see no plots.')
    parser.add_argument('-sm', '--save-model', action='store_true', default=True,
                        help='Save the model in the res folder.')
    parser.add_argument('-nd', '--num-demonstrations', type=int, default=7,
        help='Number of demonstrations from the original dataset.')
    parser.add_argument('-sd', '--save-dir', type=str,
        default=os.path.join(os.pardir, 'res', 'seds_policy'))
    parser.add_argument('-mn', '--model-name', type=str, default='test',
        help='Select a model name for the saved files.')
    parser.add_argument('-ms', '--motion-shape', type=str, default='Sine',
        help='Shape of the trajectory picked from https://bitbucket.org/khansari/lasahandwritingdataset/src/master/DataSet/')
    args = parser.parse_args()

    if args.synthetic_demo:
        learn_seds_synthetic_data(args.show_plots)

    elif args.plot_demo != "":
        ds = SE_DS()
        motion_shape=args.motion_shape
        ds.load(model_name=args.plot_demo, dir=os.path.join(os.pardir, 'res', 'seds_policy'))

        positions_py, _ = load_pylasa_data(motion_shape=motion_shape)
        plot_ds_stream(ds, positions_py, None, file_name=f'{motion_shape}_seds',
                       save_dir=args.save_dir, show_arrows=False, show_rollouts=True, show_legends=True)

    else:
        learn_seds_policy(args.motion_shape, show_plots=args.show_plots, save=args.save_model,
                          model_name=args.model_name, save_dir=args.save_dir,
                          n_dems=args.num_demonstrations)


if __name__ == '__main__':
    main()
