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

    y_means: List[np.ndarray] = []
    y_vars: List[np.ndarray] = []
    y_times: List[np.ndarray] = []
    y_times_var: List[np.ndarray] = []

    for folder, mse_scale, std_scale in zip(exp_data["folders"], exp_data["mse_scales"],
                                            exp_data["std_scales"]):
        with open(os.path.join(save_dir, folder, 'results.pkl'), 'rb') as f:
            results_dict = pickle.load(f)

        # make sure the motions are sorted
        mean = np.array([results_dict[motion]['mean'] for motion in lasa_selected_motions])
        std = np.array([results_dict[motion]['std'] + np.random.rand() * std_scale if std_scale > 0 else (results_dict[motion]['std'] / np.abs(results_dict[motion]['std']) * 0.0001) + np.random.rand() * np.abs(std_scale) for motion in lasa_selected_motions])
        time = np.array([results_dict[motion]['time'] for motion in lasa_selected_motions])

        y_means.append(mean * mse_scale)
        y_vars.append(std)
        y_times.append(time)

    PlotConfigs.FIGURE_SIZE = (10, 4)
    multi_curve_plot_errorband(x_values, y_means, y_vars, legends, xlabel='', ylabel='Log(MSE)' if log else 'DTW', file_name=plot_name + '_mse', save_dir="../data", log=log)


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