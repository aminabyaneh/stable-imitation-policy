#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

import argparse
import numpy as np

from typing import Tuple
from functools import partial

sys.path.append(os.path.join(os.pardir, 'src'))

from utils.utils import mse, time_stamp
from learn_ply_ds import PLY_SOS_DS
from utils.log_config import logger
from utils.plot_tools import plot_ds_stream, plot_contours
from policy_interface import PlanningPolicyInterface
from utils.data_loader import load_pylasa_data, lasa_selected_motions


def learn_plyds_policy(plyds_deg: int = 2, lpf_deg: int = 2,
                       motion_shape: str = "G", model_name: str = 'test',
                       optimizer: str = 'cvxpy', n_dems: int = 7, simplify_lpf : bool = True,
                       plot: bool = False, save: bool = False,
                       save_dir : str = os.path.join(os.pardir, 'res', 'plyds_policy'),
                       tol: float or int = 0.2, drop_out: float = 0.4) -> Tuple[float, PlanningPolicyInterface]:
    """ Learning sequence for a polynomial function to estimate a nonlinear dynamical
    system with Lyapunov stability.

    Args:
        plyds_deg (int, optional): Maximum degree of the dynamical system. Defaults to 2.
        lpf_deg (int, optional): Maximum complexity of lyapunov potential function. Defaults to 2.
        mode (str, optional): Switch between train and test modes. Defaults to 'train'.
        motion_shape (str, optional): Shape of the motion to load from Lasa dataset. Defaults to "G".
        model_name (str, optional): Name of the model to be saved. Defaults to 'test'.
        simplify_lpf (bool, optional): Switch on to use the most simple quadratic LPF.
        plot (bool, optional): Choose to show plots or not. Defaults to False.
        save (bool, optional): Save the model. Defaults to False.

    Returns:

        Tuple[float, PlanningPolicyInterface]: the resulting mse after the learning process,
            and the model itself
    """

    ''' Load a motion from lasa dataset '''
    model_name = model_name.lower()
    name = f'{model_name}-{motion_shape.lower()}{plyds_deg}-{time_stamp()}'

    trajectories_py, velocities_py = load_pylasa_data(motion_shape=motion_shape, n_dems=n_dems)
    logger.info(f'Handwriting dataset loaded with [{trajectories_py.shape}, {velocities_py.shape}] samples.')

    ''' Train and save a model'''
    plyds = PLY_SOS_DS(max_deg_ply=plyds_deg, max_deg_lpf=lpf_deg, drop_out=drop_out)
    plyds.fit(trajectories_py, velocities_py, optimizer=optimizer, quadratic_lpf=simplify_lpf,
              simplify_lpf=simplify_lpf, tol=tol)

    ds, lpf, dlpf_dt = plyds.get_main_functions()
    if save:
        plyds.save(model_name=name, dir=save_dir)

    ''' Test on original trajectories only '''
    trajectories_py, velocities_py = load_pylasa_data(motion_shape=motion_shape, n_dems=n_dems)
    preds = plyds.predict(trajectories_py)
    mse_val = mse(preds, velocities_py)
    logger.info(f'Final MSE on {motion_shape} data: {mse_val:.4f}')

    ''' Plot the DS '''
    if plot:
        plot_ds_stream(plyds, trajectories_py, save_dir=save_dir, file_name=f'ds-{name}',
                       show_legends=False)

    return mse_val, plyds


def main():
    """ Main entry point and argument parser for the exp file.
    """

    parser = argparse.ArgumentParser(description='Polynomial DS experiments CLI interface.')
    parser.add_argument('-dsd', '--ds-degree', type=int, default=2,
        help='Complexity of the polynomial dynamical system.')
    parser.add_argument('-lpfd', '--lpf-degree', type=int, default=2,
        help='Complexity of the stability Lyapunov function.')
    parser.add_argument('-o', '--optimizer', type=str, default="cvxpy",
        help='Switch between scipy and cvxpy optimizers.')
    parser.add_argument('-nd', '--num-demonstrations', type=int, default=50,
        help='Number of additional demonstrations to the original dataset.')
    parser.add_argument('-ms', '--motion-shape', type=str, default="G",
        help=f'Shape of the trajectories as in LASA dataset, pick from {lasa_selected_motions}.')
    parser.add_argument('-mn', '--model-name', type=str, default="test",
        help='Name of the trained model.')
    parser.add_argument('-sp', '--show-plots', action='store_true', default=False,
        help='Show extra plots of final result and trajectories.')
    parser.add_argument('-st', '--set-tolerance', type=int, default=8,
        help='Number of trials per demonstrated motion.')
    parser.add_argument('-sm', '--save-model', action='store_true', default=False,
        help='Keep a copy of the model in the res folder.')
    parser.add_argument('-sd', '--save-dir', type=str,
        default=os.path.join(os.pardir, 'res', 'plyds_policy'),
        help='Optional destination for save/load.')
    parser.add_argument('-ls', '--lpf-simplification', action='store_true', default=True,
        help='Simplify to non-parametric LPF if needed.')
    parser.add_argument('-do', '--drop-out', type=float, default=0.4,
        help='Optional destination for save/load.')

    args = parser.parse_args()
    learn_plyds_policy(plyds_deg=args.ds_degree, lpf_deg=args.lpf_degree,
        motion_shape=args.motion_shape, model_name=args.model_name,
        optimizer=args.optimizer, simplify_lpf=args.lpf_simplification, plot=args.show_plots,
        save=args.save_model, n_dems=args.num_demonstrations, tol=args.set_tolerance,
        save_dir=args.save_dir, drop_out=args.drop_out)


if __name__ == '__main__':
    main()
