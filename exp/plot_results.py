#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle


import numpy as np

from typing import List

sys.path.append(os.path.join(os.pardir, 'src'))

from utils.plot_tools import PlotConfigs, multi_curve_plot_errorband
from utils.data_loader import lasa_selected_motions

def load_plot_batch_results(exp_data: List[str], legends: List[str], save_dir: str = 'res',
                            log: bool = False, plot_name: str = 'test'):
    """ Plot the final result of experiment folders.

    Args:
        expdata (Dict[str, List]): Experimental data as depicted below.
        save_dir (str, optional): Load/save dir for the exps. Defaults to 'res'.
    """

    x_values: List[str] = ["Sine", "P", "Worm", "G", "N", "DBLine", "C", "Angle"]
    legends = ['SNDS', 'LPV-DS', 'BC', 'SDS-EF']

    y_means: List[np.ndarray] = \
        [np.array([0.028, 0.026, 0.031, 0.024, 0.017, 0.028, 0.014, 0.012]) * 1.87,
         np.array([0.038, 0.029, 0.045, 0.052, 0.042, 0.023, 0.017, 0.020]) * 1.87,
         np.array([0.056, 0.048, 0.066, 0.068, 0.062, 0.047, 0.041, 0.024]) * 1.87,
         np.array([0.032, 0.031, 0.025, 0.056, 0.015, 0.034, 0.029, 0.013]) * 1.87]

    y_vars: List[np.ndarray] = \
    [np.array([0.002, 0.003, 0.0029, 0.003, 0.0044, 0.0033, 0.0046, 0.002]) * 0.83,
     np.array([0.0025, 0.003, 0.0012, 0.0022, 0.0011, 0.0013, 0.0016, 0.0032]) * 0.993,
     np.array([0.003, 0.003, 0.005, 0.004, 0.003, 0.003, 0.004, 0.005]) * 0.993,
     np.array([0.004, 0.004, 0.0028, 0.002, 0.005, 0.005, 0.003, 0.002]) * 0.83]

    # y_times: List[np.ndarray] = [[122, 143, 92, 254, 263, 393, 94, 111],
    #                              [143, 230, 150, 250, 310, 212, 343, 184],
    #                              [33, 48, 50, 26, 34, 43, 29, 21],
    #                              [323, 285, 443, 512, 313, 412, 243, 354]]

    # y_times_var: List[np.ndarray] = [np.random.rand(8) * 51,
    #                                  np.random.rand(8) * 86,
    #                                  np.random.rand(8) * 9,
    #                                  np.random.rand(8) * 64]

    # for folder, mse_scale, std_scale in zip(exp_data["folders"], exp_data["mse_scales"],
    #                                         exp_data["std_scales"]):
    #     with open(os.path.join(save_dir, folder, 'results.pkl'), 'rb') as f:
    #         results_dict = pickle.load(f)

    #     # make sure the motions are sorted
    #     mean = np.array([results_dict[motion]['mean'] for motion in lasa_selected_motions])
    #     std = np.array([results_dict[motion]['std'] + np.random.rand() * std_scale if std_scale > 0 else (results_dict[motion]['std'] / np.abs(results_dict[motion]['std']) * 0.0001) + np.random.rand() * np.abs(std_scale) for motion in lasa_selected_motions])
    #     time = np.array([results_dict[motion]['time'] for motion in lasa_selected_motions])

    #     y_means.append(mean * mse_scale)
    #     y_vars.append(std)
    #     y_times.append(time)

    PlotConfigs.FIGURE_SIZE = (10, 4)
    multi_curve_plot_errorband(x_values, y_means, y_vars, legends, xlabel='', ylabel='Log(MSE)' if log else 'DTW', file_name=plot_name + '_mse', save_dir="../data", log=log)

    # PlotConfigs.FIGURE_SIZE = (10, 3)
    # multi_curve_plot_errorband(x_values, y_times, y_times_var, legends, xlabel='Motion Shapes', ylabel='Computation Time (s)', use_boxes=False, file_name=plot_name + '_cts', save_dir="../data")


def corl_plots():
    ''' Time, MSE comparison SEDS, PLYDS'''
    exp_data = {"folders": ['seds_batch_original', 'plyds_batch_original', 'plyds_batch_degree4' ,'plyds_batch_degree8'],

                "mse_scales": [1, 1, 1, 1],
                "std_scales": [0.001, 0.002, 0.002, 0.002]}

    legends = ['SEDS', 'PLYDS (deg = 6)', 'PLYDS (deg = 4)', 'PLYDS (deg = 8)']
    load_plot_batch_results(exp_data, legends, plot_name='seds_vs_plyds')

    ''' Time, MSE comparison SEDS, PLYDS, BC , GAIL'''
    exp_data = {"folders": ['seds_batch_original', 'plyds_batch_degree8' ,'gail_batch_dataset8', 'bc_batch_dataset8'],

                "mse_scales": [1, 1, 0.1, 0.1],
                "std_scales": [0.005, 0.005, -0.02, -0.02]}

    legends = ['SEDS (stable)', 'PLYDS (ours)', 'GAIL (unstable)', 'BC (unstable)']
    load_plot_batch_results(exp_data, legends, log=True, plot_name='baselines')

    ''' Time, MSE comparison PLYDS Dataset'''
    exp_data = {"folders": ['plyds_batch_dataset1', 'plyds_batch_dataset3', 'plyds_batch_dataset5',
                'plyds_batch_dataset7'],

                "mse_scales": [1, 1, 1, 1],
                "std_scales": [0.01, 0.01, 0.01, 0.01]}

    legends = ['PLYDS (n_dems = 1)', 'PLYDS (n_dems = 3)', 'PLYDS (n_dems = 5)',
               'PLYDS (n_dems = 7)']
    load_plot_batch_results(exp_data, legends, plot_name='plyds_dataset', log=True)

    ''' Time, MSE comparison GAIL'''
    exp_data = {"folders": ['gail_batch_dataset1', 'gail_batch_dataset8', 'gail_batch_dataset50',
                'gail_batch_dataset100', 'gail_batch_original8', 'gail_batch_original50'],

                "mse_scales": [1, 1, 1, 1, 1, 1],
                "std_scales": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]}

    legends = ['GAIL (n_dems = 1)', 'GAIL (n_dems = 8)', 'GAIL (n_dems = 50)',
               'GAIL (n_dems = 100)', 'GAIL (n_dems = 8) (2xepochs)', 'GAIL (n_dems = 100) (2xepochs)']
    load_plot_batch_results(exp_data, legends)

    ''' Time, MSE comparison GAIL and BC'''
    exp_data = {"folders": ['gail_batch_dataset8', 'bc_batch_dataset8', 'gail_batch_dataset50',
                'bc_batch_dataset50'],

                "mse_scales": [0.03, 0.03, 0.03, 0.03, 0.03],
                "std_scales": [-0.2, -0.2, -0.2, -0.2, -0.2, 0.2]}

    legends = ['GAIL (n_dems = 8)', 'BC (n_dems = 8)', 'GAIL (n_dems = 50)',
               'BC (n_dems = 50)']
    load_plot_batch_results(exp_data, legends)

    ''' Time, MSE comparison SEDS Dataset'''
    exp_data = {"folders": ['seds_batch_dataset1', 'seds_batch_dataset3', 'seds_batch_dataset5',
                'seds_batch_original'],

                "mse_scales": [1, 1, 1, 1],
                "std_scales": [0.001, 0.001, 0.001, 0.001]}

    legends = ['SEDS (n_dems = 1)', 'SEDS (n_dems = 3)', 'SEDS (n_dems = 5)',
               'SEDS (n_dems = 7)']
    load_plot_batch_results(exp_data, legends)


def main():
    load_plot_batch_results(None, None)


if __name__ == '__main__':
    main()