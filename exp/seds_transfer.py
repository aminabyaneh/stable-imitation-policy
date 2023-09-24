#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.pardir, 'src'))

from typing import List
from utils.log_config import logger
from utils.transfer_tools import evaluate_transfer_retrain
from utils.plot_tools import plot_performance_curves, plot_trajectory

from utils.data_loader import generate_synthetic_linear_data, load_pylasa_data
from utils.data_loader import N_DEMONSTRATIONS_LASA_HANDWRITING, N_SAMPLES_LASA_HANDWRITING


def linear_direct_ds_transfer(show_ds_plots: bool = True, num_execs: int = 1) -> List:
    """ Utilizing synthetic data, see if fine-tuning works as
    a TF learning approach for learning DS.

    Args:
        show_ds_plots (bool, optional): Whether to show plots for learned ds.
            Defaults to True.
        num_execs (int, optional): Number of seeds to run the experiment for.
            Defaults to 1.
    """

    ''' Synthetic data setup '''
    ds_a_mat = np.matrix([[-1., 1.], [1., -10.]], dtype=np.float64)

    n_samples = N_SAMPLES_LASA_HANDWRITING
    n_demonstrations = N_DEMONSTRATIONS_LASA_HANDWRITING

    ''' Train, test, and limit splits '''
    n_test_dem = 1
    n_train_dem = n_demonstrations - n_test_dem
    n_limit_dem = 1

    ''' Multiple executions for error bands'''
    results = list()
    for ـ in range(num_execs):

        ''' Synthetic data generation '''
        syn_pos, syn_vel = generate_synthetic_linear_data(A=ds_a_mat, start_point=(10., -15.),
            target_point=(0., 0.), n_dems=n_demonstrations, n_samples=n_samples)
        motions_data = {"linear": (syn_pos, syn_vel)}

        ''' Evaluate the direct transfer '''
        res = evaluate_transfer_retrain(motions_data, {"linear": "linear"}, n_train_dem,
            n_test_dem, n_limit_dem, n_samples, is_linear=True, show_plots=show_ds_plots)
        results.append(res)


    ''' Saving and printing '''
    res_path = os.path.join(os.pardir, "res", 'linear_direct_transfer.npy')

    np.save(res_path, results)
    plot_performance_curves(file_path=res_path)


def linear_ds_indirect_transfer(show_ds_plots: bool = True, num_execs: int = 1):
    """ Utilizing synthetic data, see if fine-tuning works as
    a TF learning approach for learning DS.

    Args:
        show_ds_plots (bool, optional): Whether to show plots for learned ds.
            Defaults to True.
        num_execs (int, optional): Number of seeds to run the experiment for.
            Defaults to 1.
    """

    ''' Synthetic data generation '''
    ds_src_a_mat = np.matrix([[-1., 0.], [0., -1.]], dtype=np.float64)
    ds_trg_a_mat = np.array([[-5., 2.], [1., -4.]], dtype=np.float64)

    n_samples = N_SAMPLES_LASA_HANDWRITING // 100
    n_dems = N_DEMONSTRATIONS_LASA_HANDWRITING

    ''' Train, test, and limit splits '''
    n_test_dem = 1
    n_train_dem = n_dems - n_test_dem
    n_limit_dem = 1

    ''' Multiple executions for error bands'''
    results = list()
    for ـ in range(num_execs):

        ''' Synthetic data generation '''
        syn_pos_src, syn_vel_src = generate_synthetic_linear_data(A=ds_src_a_mat,
            start_point=(10., -15.), target_point=(0., 0.), n_dems=n_dems,
            n_samples=n_samples)

        syn_pos_trg, syn_vel_trg = generate_synthetic_linear_data(A=ds_trg_a_mat,
            start_point=(10., -15.), target_point=(0., 0.), n_dems=n_dems,
            n_samples=n_samples)

        motions_data = {"linear_src": (syn_pos_src, syn_vel_src), "linear_trg": (syn_pos_trg, syn_vel_trg)}
        transfer_map = {"linear_src": "linear_trg", "linear_trg": "linear_src"}

        if show_ds_plots:
            plot_trajectory(syn_pos_src, 'Synthetic trajectories for source linear DS')
            plot_trajectory(syn_pos_trg, 'Synthetic trajectories for target linear DS')

        ''' Evaluate the indirect transfer '''
        res = evaluate_transfer_retrain(motions_data, transfer_map, n_train_dem, n_test_dem,
            n_limit_dem, n_samples, is_linear=True, show_plots=show_ds_plots)

        results.append(res)

    ''' Saving and printing '''
    res_path = os.path.join(os.pardir, "res", 'linear_indirect_transfer.npy')

    np.save(res_path, results)
    plot_performance_curves(file_path=res_path, keys=motions_data.keys())


def nonlinear_ds_direct_transfer(show_ds_plots: bool = True, num_execs: int = 1):
    """ Utilizing data for sine, C, and G motion, see if a warm start has
    any positive effects when using a reference DS of the same trajectory.

    Args:
        show_ds_plots (bool, optional): Whether to show plots for learned ds.
            Defaults to True.
        num_execs (int, optional): Number of seeds to run the experiment for.
            Defaults to 1.
    """

    ''' Load data sine, g, and c motions '''
    n_samples = N_SAMPLES_LASA_HANDWRITING
    n_dems = N_DEMONSTRATIONS_LASA_HANDWRITING

    s_pos, s_vel = load_pylasa_data(motion_shape='Sine')
    g_pos, g_vel = load_pylasa_data(motion_shape='G')
    c_pos, c_vel = load_pylasa_data(motion_shape='C')

    n_test_dem = 1
    n_train_dem = n_dems - n_test_dem
    n_limit_dem = 1

    motions_data = {"S": (s_pos, s_vel), "C": (c_pos, c_vel), "G": (g_pos, g_vel)}
    transfer_map = {"S": "S", "C": "C", "G": "G"}

    ''' Multiple executions for error bands'''
    results = list()
    for ـ in range(num_execs):

        if show_ds_plots:
            for shape in motions_data:
                plot_trajectory(motions_data[shape], f'Synthetic trajectories for {shape}-shaped DS')

        ''' Evaluate the indirect transfer '''
        res = evaluate_transfer_retrain(motions_data, transfer_map, n_train_dem, n_test_dem,
            n_limit_dem, n_samples, is_linear=False, show_plots=show_ds_plots)

        results.append(res)

    ''' Saving and printing '''
    res_path = os.path.join(os.pardir, "res", 'nonlinear_direct_transfer.npy')

    np.save(res_path, results)
    plot_performance_curves(file_path=res_path, keys=motions_data.keys())


def main():
    """ Main entry point for the exp file.
    """

    parser = argparse.ArgumentParser(description='Transfer DS experiments CLI interface.')
    parser.add_argument('--linear', action='store_true', default=False,
        help='Experiments conducted assuming a linear DS as the data generation mechanism.')
    parser.add_argument('--nonlinear', action='store_true', default=False,
        help='Experiments for nonlinear DS using lasa dataset data instead of synthetic data.')
    parser.add_argument('--direct', action='store_true',
        help='Whether to use direct or indirect transfer. The direct case applies the transfer to '
        f'same source and target and is only used as a sanity check for warm initialization', default=False)
    parser.add_argument('--show-gmm-plots', action='store_true',
        help='Set false to see no GMM plots.', default=False)
    parser.add_argument('--show-ds-plots', action='store_true',
        help='Set false to see no DS plots.', default=False)
    parser.add_argument('--num-execs', type=int, help='Number of executions or different seeds.', default=1)
    args = parser.parse_args()

    logger.info(f'Executing a direct: {args.direct} and linear: {args.linear} transfer effort.')
    if args.direct:
        if args.linear:
            linear_direct_ds_transfer(args.show_ds_plots, args.num_execs)
        elif args.nonlinear:
            nonlinear_ds_direct_transfer(args.show_ds_plots, args.num_execs)
        else:
            logger.error('Choose either the --linear or --nonlinear option.')
    else:
        if args.linear:
            linear_ds_indirect_transfer(args.show_ds_plots, args.num_execs)
        else:
            logger.error('Option(s) not implemented, please adjust your entries.')
            pass

if __name__ == '__main__':
    main()